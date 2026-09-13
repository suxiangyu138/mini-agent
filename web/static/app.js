/* Mini-Agent 前端
 *
 * 无框架、无构建，一个文件读完。接口分四组：
 *   GET  /api/info                     界面启动时问一次当前配置和工具
 *   POST /api/chat                     提问，响应是 SSE 流（见 handleFrame）
 *   POST /api/cancel                   中断正在跑的那一轮
 *   GET  /api/sessions /api/session /api/context /api/memories /api/settings
 *   POST /api/sessions/{new,select,rename,pin,delete,clear}
 *        /api/memories/{add,update,delete,clear} · /api/settings · /api/reset
 *
 * 两条硬规则：
 *
 * 1. **模型输出的文字一律先转义再进 DOM**。模型会把 http_request 抓回来的网页
 *    内容当上下文，那里面写什么都有可能，不转义就等于把 XSS 直接送到页面上。
 * 2. **一条 SSE 帧只能写进它自己那一轮**。done 之后输入框就解锁了，用户完全
 *    可能在上一轮的收尾事件（记忆 / 压缩 / 上下文）还在路上时就开始下一轮；
 *    渲染函数因此一律带 turn 参数，不认「当前那一轮」这个隐式状态。
 */

(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);

  const el = {
    app: $('app'),
    stream: $('stream'),
    turns: $('turns'),
    empty: $('empty'),
    suggestions: $('suggestions'),
    input: $('input'),
    send: $('send'),
    stop: $('stop'),
    chip: $('model-chip'),
    modelText: $('model-text'),
    sheet: $('sheet'),
    toolList: $('tool-list'),
    // 侧栏与各页
    sideList: $('side-list'),
    sideTabs: document.querySelectorAll('.side-tab'),
    crumb: $('crumb'),
    notice: $('notice'),
    noticeText: $('notice-text'),
    pageMemory: $('page-memory'),
    pageSettings: $('page-settings'),
    scrim: $('scrim'),
    // 上下文状态
    ctx: $('ctx'),
    ctxLine: $('ctx-line'),
    ctxText: $('ctx-text'),
    ctxBar: $('ctx-bar'),
    ctxDetail: $('ctx-detail'),
    // 浮层
    toastHost: $('toast-host'),
    dialog: $('dialog'),
    dialogPanel: $('dialog-panel'),
  };

  const state = {
    info: { tools: [], suggestions: [], settings: {} },
    sessionId: '', // 页面上开着的那个会话。每一轮都带上它，避免写进别的会话
    running: false, // 正在跑一轮
    pinned: true, // 用户是否还停在底部（自己往上翻了就别硬拽回来）
    view: 'chat', // chat | memory | settings
    sessions: { current: '', count: 0, groups: [] },
    context: null, // 最近一次 /api/context 的结果
    noticeTimer: 0,
  };

  // ==================================================================== 工具函数

  const money = (n) => n.toLocaleString('en-US');
  const fmt = (digits, value) => Number(value).toFixed(digits);

  function span(cls, text) {
    const node = document.createElement('span');
    if (cls) node.className = cls;
    if (text !== undefined && text !== null) node.textContent = text;
    return node;
  }

  /** 一个指标格：`标签 数值 单位`。指标条的每个格子都长这样，三处渲染共用。 */
  function metric(label, value, unit, cls) {
    const node = span('m' + (cls ? ' ' + cls : ''));
    if (label) node.appendChild(span('unit', label));
    const strong = document.createElement('b');
    strong.textContent = value;
    node.appendChild(strong);
    if (unit) node.appendChild(span('unit', unit));
    return node;
  }

  const stamp = () =>
    new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });

  // ==================================================================== 文本渲染
  //
  // 分三层：**块级 → 行内 → 公式**。
  //
  // 一条贯穿始终的硬规则：模型输出的文字一律先转义再进 DOM。
  // 全文件只有一处 innerHTML（在 inlineHtml 里），它能成立是因为进去的字符串
  // 已经过 escapeHtml，之后只加回 <code>/<strong>/<a> 三种自己写死的标签。
  // 公式刻意不走这条路 —— KaTeX 直接往 DOM 节点里写，不产生 HTML 字符串，
  // 所以「反正是 KaTeX 生成的」这种信任不需要建立。
  //
  // 不做完整 CommonMark，只认模型真会写出来的那几种。
  // 但**容错优先于严格**：围栏没闭合、表格列数不齐、语言名写成 py，
  // 都要渲染出一个说得过去的结果，而不是把后面的内容整段吞掉。

  const ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ESCAPES[c]);

  /** 行内标记：代码、加粗、链接。入参是**原始文本**，转义在这一步做。 */
  function inlineHtml(raw) {
    // 先按行内代码切开：代码段原样留着，其余部分才做加粗和链接。
    // 不切的话 `**x**` 会被后面的加粗规则二次加工成 <code><strong>x</strong></code>。
    return escapeHtml(raw)
      .split(/(`[^`\n]+`)/)
      .map((part) =>
        part.length > 2 && part.startsWith('`') && part.endsWith('`')
          ? '<code>' + part.slice(1, -1) + '</code>'
          : part
              .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
              .replace(
                /(https?:\/\/[^\s<>()]+)/g,
                '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
              )
      )
      .join('')
      .replace(/\n/g, '<br>');
  }

  // ---- 公式 ----
  //
  // 四种定界符，两种含义：$...$ 和 \(...\) 是行内公式，$$...$$ 和 \[...\] 独立成行。
  //
  // 最容易误判的是**货币符号**：模型答「约合 670.82 元」时写 $100 是常事，
  // 一旦把那个 $ 当成公式开头，后面半段话会被整个吞进去。
  // 所以 $ 后面紧跟数字或空白的一律不当公式开头。

  /** 找 from 之后 closer 的位置；allowNewline 为假时不跨行。 */
  function closeAt(text, from, closer, allowNewline) {
    const at = text.indexOf(closer, from);
    if (at < 0 || allowNewline) return at;
    const newline = text.indexOf('\n', from);
    return newline >= 0 && newline < at ? -1 : at;
  }

  /** 行内公式的收尾 $：不能跨行，前面不能是空格，后面不能还是 $。 */
  function inlineEnd(text, from) {
    for (let i = from; i < text.length; i++) {
      if (text[i] === '\n') return -1; // 不跨行，否则落单的 $ 能吞掉半个回答
      if (text[i] === '\\') i++; // \$ 是字面量
      else if (text[i] === '$' && !/\s/.test(text[i - 1]) && text[i + 1] !== '$') return i;
    }
    return -1;
  }

  /**
   * 把原始文本切成「普通文本」和「公式」交替的片段。
   * 从左到右扫一遍，每个位置按**长定界符优先**试（$$ 要先于 $ 判）。
   */
  function splitMath(raw) {
    const text = String(raw);
    const pieces = [];
    let plain = '';
    let i = 0;

    const flush = () => {
      if (plain) pieces.push({ text: plain });
      plain = '';
    };

    while (i < text.length) {
      let tex = null;
      let display = false;
      let end = i;

      // 行内代码里的 $ 是字面量，整段原样跳过 —— 不跳的话 `$x$` 会被渲染成公式，
      // 而代码的语义恰恰是「照原样显示」。
      if (text[i] === '`') {
        const close = text.indexOf('`', i + 1);
        const newline = text.indexOf('\n', i + 1);
        if (close > i && (newline < 0 || close < newline)) {
          plain += text.slice(i, close + 1);
          i = close + 1;
          continue;
        }
      }

      if (text.startsWith('$$', i)) {
        // 独立公式可以跨多行 —— 模型经常写成 $$\n...\n$$，中间那几个换行要一起吃掉
        const close = closeAt(text, i + 2, '$$', true);
        if (close > i + 2) {
          tex = text.slice(i + 2, close);
          display = true;
          end = close + 2;
        }
      } else if (text.startsWith('\\[', i)) {
        const close = closeAt(text, i + 2, '\\]', true);
        if (close > i + 2) {
          tex = text.slice(i + 2, close);
          display = true;
          end = close + 2;
        }
      } else if (text.startsWith('\\(', i)) {
        const close = closeAt(text, i + 2, '\\)', false);
        if (close > i + 2) {
          tex = text.slice(i + 2, close);
          end = close + 2;
        }
      } else if (text[i] === '$' && text[i - 1] !== '\\') {
        const next = text[i + 1];
        // 货币符号保护：$ 后面是空白或又是 $ 的，一律不是公式开头
        const blocked = next === undefined || next === '$' || /\s/.test(next);
        if (!blocked) {
          const close = inlineEnd(text, i + 1);
          if (close > i + 1) {
            const body = text.slice(i + 1, close);
            // $ 后面紧跟数字，绝大多数时候是金额（$100）。但也有真的以数字
            // 开头的公式（$778{,}516.9$）。两者用「内容里有没有 LaTeX 记号」分：
            // 金额里不会出现 \ { } ^ _，而公式里几乎总有。
            if (!/\d/.test(next) || /[\\{}^_]/.test(body)) {
              tex = body;
              end = close + 1;
            }
          }
        }
      }

      const body = tex === null ? '' : tex.trim();
      if (!body) {
        // 不是公式开头。\x 整体当字面量跳过，免得 \$ 里那个 $ 又被当成定界符。
        if (text[i] === '\\' && i + 1 < text.length) {
          plain += text.slice(i, i + 2);
          i += 2;
        } else {
          plain += text[i];
          i += 1;
        }
        continue;
      }

      flush();
      pieces.push({ tex: body, display });
      i = end;
    }

    flush();
    return pieces;
  }

  /**
   * 一个公式 → 一个 DOM 节点。**不走 innerHTML**：KaTeX 自己往节点里写。
   *
   * 渲染失败一律降级成源码：宁可让用户看见 \frac{1}{2} 这样的原文，
   * 也不能显示一堆红字或者干脆空白 —— 原文至少还读得懂，空白就什么信息都没了。
   * KaTeX 没加载出来（离线、CDN 被拦）走的是同一条降级路径。
   */
  function mathNode(tex, display) {
    const node = document.createElement('span');
    node.className = display ? 'math math-display' : 'math';

    if (window.katex && typeof window.katex.render === 'function') {
      try {
        window.katex.render(tex, node, {
          displayMode: display,
          throwOnError: true, // 出错要抛出来，抛了我们才接得住、才好降级
          strict: false, // 模型写点非标准 LaTeX 也认，别动不动就报错
          trust: false, // 关掉 \href / \htmlClass 这类能往属性里塞东西的命令
          maxSize: 12, // 挡住 \Huge 之类把行高撑爆的尺寸命令
        });
        node.dataset.tex = tex; // 悬停看得见原始 LaTeX，方便复制去别处
        return node;
      } catch (error) {
        node.classList.add('math-error');
        node.title = '公式渲染失败：' + (error && error.message ? error.message : error);
      }
    } else {
      node.classList.add('math-error');
      node.title = 'KaTeX 没加载成功（离线或 CDN 被拦），这里显示的是公式源码';
    }

    node.textContent = (display ? '$$' : '$') + tex + (display ? '$$' : '$');
    return node;
  }

  /** 行内内容 → DOM 片段：文本走「转义 + 白名单标签」，公式交给 KaTeX。 */
  function inlineNodes(raw) {
    const frag = document.createDocumentFragment();
    for (const piece of splitMath(raw)) {
      if (piece.tex === undefined) {
        const holder = document.createElement('span');
        holder.innerHTML = inlineHtml(piece.text);
        while (holder.firstChild) frag.appendChild(holder.firstChild);
        continue;
      }
      frag.appendChild(mathNode(piece.tex, piece.display));
    }
    return frag;
  }

  // ---- 块级 ----

  const FENCE_OPEN = /^\s{0,3}(`{3,}|~{3,})\s*(.*?)\s*$/;
  const FENCE_CLOSE = /^\s{0,3}(`{3,}|~{3,})\s*$/;
  const INFO_TOKEN = /^[A-Za-z0-9_+#.-]{1,20}/;
  const HEADING = /^\s{0,3}(#{1,6})\s+(.*?)\s*#*\s*$/;
  const HR = /^\s{0,3}([-*_])\s*(?:\1\s*){2,}$/;
  const QUOTE = /^\s{0,3}>\s?(.*)$/;
  //: 列表标记单独捕获成一组：renderList 要拿它算正文缩进在哪一列。
  //: 两个正则的组位置因此是平行的：1=缩进 2=标记 3=标记后的空白 4=正文。
  const BULLET = /^(\s*)([-*+])(\s+)(.*)$/;
  const NUMBERED = /^(\s*)(\d{1,9})([.)])(\s+)(.*)$/;
  //: 表格分隔行：|---|:--:|---|，两头的竖线可省
  const TABLE_SEP = /^\s*\|?\s*:?-{1,}:?\s*(?:\|\s*:?-{1,}:?\s*)*\|?\s*$/;

  //: 语言别名归一化。模型写 py / js / sh 都很常见，统一成规范名，
  //: 代码块右上角那个标签才不至于一会儿 py 一会儿 python。
  const LANG_ALIAS = {
    py: 'python', python3: 'python', python2: 'python',
    js: 'javascript', node: 'javascript', mjs: 'javascript', cjs: 'javascript',
    ts: 'typescript', sh: 'bash', shell: 'bash', zsh: 'bash', console: 'bash',
    yml: 'yaml', md: 'markdown', rs: 'rust', kt: 'kotlin', golang: 'go',
    'c++': 'cpp', cxx: 'cpp', 'c#': 'csharp', cs: 'csharp', rb: 'ruby',
    ps1: 'powershell', htm: 'html', text: 'text', txt: 'text',
  };

  /** 围栏代码块。语言名只取首token，后面跟的 title="..." 之类直接忽略。 */
  function codeBlock(body, lang) {
    const raw = (lang || '').toLowerCase();
    const name = LANG_ALIAS[raw] || raw;

    const pre = document.createElement('pre');
    const code = document.createElement('code');
    code.textContent = body;
    if (name) {
      code.className = 'language-' + name;
      pre.dataset.lang = name; // 右上角那个小标签读它（见 style.css）
    }
    pre.appendChild(code);
    return pre;
  }

  /** 按未转义的 | 切单元格；\| 是格子里的字面竖线。 */
  function splitRow(line) {
    const text = line.trim().replace(/^\||\|$/g, '');
    const cells = [];
    let cell = '';
    for (let i = 0; i < text.length; i++) {
      if (text[i] === '\\' && text[i + 1] === '|') {
        cell += '|';
        i++;
      } else if (text[i] === '|') {
        cells.push(cell);
        cell = '';
      } else {
        cell += text[i];
      }
    }
    cells.push(cell);
    return cells.map((c) => c.trim());
  }

  /**
   * 表格。分隔行决定每列的对齐。
   * **列数不齐是常态**（模型经常少写或多写一格），所以一律以表头列数为准：
   * 多的截掉、少的补空格子，绝不让后面的行整体错位。
   */
  function renderTable(frag, lines, start) {
    const head = splitRow(lines[start]);
    const aligns = splitRow(lines[start + 1]).map((cell) => {
      const left = cell.startsWith(':');
      const right = cell.endsWith(':');
      return left && right ? 'center' : right ? 'right' : left ? 'left' : '';
    });

    const table = document.createElement('table');
    const headRow = table.createTHead().insertRow();
    head.forEach((cell, index) => {
      const th = document.createElement('th');
      if (aligns[index]) th.style.textAlign = aligns[index];
      th.appendChild(inlineNodes(cell));
      headRow.appendChild(th);
    });

    const body = table.createTBody();
    let i = start + 2;
    while (i < lines.length && lines[i].trim() && lines[i].includes('|')) {
      const cells = splitRow(lines[i]);
      const row = body.insertRow();
      for (let c = 0; c < head.length; c++) {
        const td = row.insertCell();
        if (aligns[c]) td.style.textAlign = aligns[c];
        td.appendChild(inlineNodes(cells[c] === undefined ? '' : cells[c]));
      }
      i++;
    }

    // 宽表格自己横向滚，不能把整页撑出横向滚动条
    const wrap = document.createElement('div');
    wrap.className = 'table-wrap';
    wrap.appendChild(table);
    frag.appendChild(wrap);
    return i;
  }

  /**
   * 有序 / 无序列表。
   *
   * 每条的内容**反缩进后递归渲染**，所以嵌套列表、列表里放代码块、
   * 引用里套列表都不用单独写代码 —— 递归下去自然就对了。
   */
  function renderList(frag, lines, start, ordered) {
    const ITEM = ordered ? NUMBERED : BULLET;
    const list = document.createElement(ordered ? 'ol' : 'ul');

    // 从模型给的第一个序号开始（从 3 开始的列表就显示 3、4、5）
    const first = Number(ITEM.exec(lines[start])[2]);
    if (ordered && first > 1) list.start = first;

    let i = start;
    while (i < lines.length && ITEM.test(lines[i])) {
      const matched = ITEM.exec(lines[i]);
      const indent = matched[1].length;
      // 正文列 = 缩进 + 标记宽 + 一个空格；有序列表的标记宽是「数字 + 点」
      const marker = ordered
        ? matched[2].length + matched[3].length
        : matched[2].length;
      const column = indent + marker + 1;

      const body = [ordered ? matched[5] : matched[4]];
      i++;
      while (i < lines.length) {
        const line = lines[i];
        const at = line.search(/\S/);
        if (at < 0) {
          // 空行后面还跟着这个列表的内容就继续，否则列表到此为止
          const next = lines[i + 1];
          if (next === undefined || (!ITEM.test(next) && next.search(/\S/) < indent)) break;
          body.push('');
          i++;
          continue;
        }
        if (at === indent && ITEM.test(line)) break; // 同级的下一条
        if (at < column) break; // 缩进不够，列表结束
        body.push(line.slice(column));
        i++;
      }

      const li = document.createElement('li');
      li.appendChild(renderRich(body.join('\n')));
      list.appendChild(li);
    }

    frag.appendChild(list);
    return i;
  }

  /** 这一行是不是某个块的开头（段落循环靠它决定在哪断开）。 */
  function startsBlock(lines, i) {
    const line = lines[i];
    if (FENCE_OPEN.test(line) || HEADING.test(line) || HR.test(line)) return true;
    if (QUOTE.test(line) || BULLET.test(line) || NUMBERED.test(line)) return true;
    return line.includes('|') && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1]);
  }

  /** markdown → DOM 片段。一次扫一遍行，按行首特征决定这块是什么。 */
  function renderRich(text) {
    const frag = document.createDocumentFragment();
    const lines = String(text).replace(/\r\n?/g, '\n').split('\n');
    let i = 0;

    while (i < lines.length) {
      const line = lines[i];

      // ---- 围栏代码块 ----
      const open = FENCE_OPEN.exec(line);
      if (open) {
        const marker = open[1][0];
        const size = open[1].length;
        const info = INFO_TOKEN.exec(open[2]);
        const body = [];
        i++;
        // 找不到收尾就一直吃到文末：回答被 max_tokens 截断时，
        // 剩下半个代码块按代码显示才对，不该被当成正文。
        while (i < lines.length) {
          const close = FENCE_CLOSE.exec(lines[i]);
          if (close && close[1][0] === marker && close[1].length >= size) break;
          body.push(lines[i]);
          i++;
        }
        i++;
        frag.appendChild(codeBlock(body.join('\n'), info ? info[0] : ''));
        continue;
      }

      // ---- 标题 ----
      const heading = HEADING.exec(line);
      if (heading) {
        const node = document.createElement('h' + heading[1].length);
        node.appendChild(inlineNodes(heading[2]));
        frag.appendChild(node);
        i++;
        continue;
      }

      // ---- 分隔线（必须排在列表前面，否则「- - -」会被当成列表项）----
      if (HR.test(line)) {
        frag.appendChild(document.createElement('hr'));
        i++;
        continue;
      }

      if (!line.trim()) {
        i++;
        continue;
      }

      // ---- 表格 ----
      if (line.includes('|') && i + 1 < lines.length && TABLE_SEP.test(lines[i + 1])) {
        i = renderTable(frag, lines, i);
        continue;
      }

      // ---- 引用 ----
      if (QUOTE.test(line)) {
        const inner = [];
        while (i < lines.length && lines[i].trim() && (QUOTE.test(lines[i]) || inner.length)) {
          const quoted = QUOTE.exec(lines[i]);
          inner.push(quoted ? quoted[1] : lines[i]);
          i++;
        }
        const node = document.createElement('blockquote');
        node.appendChild(renderRich(inner.join('\n'))); // 引用里可以有列表和代码块
        frag.appendChild(node);
        continue;
      }

      // ---- 列表 ----
      if (BULLET.test(line)) {
        i = renderList(frag, lines, i, false);
        continue;
      }
      if (NUMBERED.test(line)) {
        i = renderList(frag, lines, i, true);
        continue;
      }

      // ---- 段落（兜底）----
      const para = [line];
      i++;
      while (i < lines.length && lines[i].trim() && !startsBlock(lines, i)) {
        para.push(lines[i]);
        i++;
      }
      const p = document.createElement('p');
      p.appendChild(inlineNodes(para.join('\n')));
      frag.appendChild(p);
    }

    return frag;
  }

  // ==================================================================== 渲染

  function atBottom() {
    return el.stream.scrollHeight - el.stream.scrollTop - el.stream.clientHeight < 120;
  }

  function scrollToEnd(force) {
    if (force || state.pinned) el.stream.scrollTop = el.stream.scrollHeight;
  }

  /**
   * 开一轮，返回这一轮的句柄。
   *
   * 句柄是**显式传下去**的，不放进 state 里当「当前那一轮」：done 之后输入框
   * 就解锁了，用户可能在上一轮的收尾事件还没到齐时就开始下一轮，而「当前那一轮」
   * 只有一个 —— 那时候上一轮的记忆卡片就会画到新一轮下面去。
   */
  function makeTurn(question) {
    el.empty.hidden = true;

    const node = document.createElement('article');
    node.className = 'turn';

    const row = document.createElement('div');
    row.className = 'msg-user';
    row.appendChild(span('stamp', stamp()));
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = question;
    row.appendChild(bubble);
    node.appendChild(row);

    const bot = document.createElement('div');
    bot.className = 'msg-bot';
    bot.setAttribute('aria-busy', 'true');
    node.appendChild(bot);

    // 流式期间有个「正在发生」的指标行垫在最底下，新内容插在它前面，
    // 这样工具块和文字块的先后顺序自然就是它们真实发生的顺序。
    const live = document.createElement('div');
    live.className = 'metrics live';
    bot.appendChild(live);

    el.turns.appendChild(node);

    const turn = {
      node,
      bot,
      live,
      prose: null, // 当前正在追加文字的段落块
      text: '', // 已经收到的正文
      chars: 0,
      startedAt: performance.now(),
      firstAt: null,
      lastPaint: 0,
      settled: false, // metrics 已经画过，后面的事件该往它下面追加
    };
    paintLive(turn);
    scrollToEnd(true);
    return turn;
  }

  function liveRow(t) {
    const now = performance.now();
    if (t.firstAt === null) t.firstAt = now;
    const generation = Math.max((now - t.firstAt) / 1000, 0);
    const total = (now - t.startedAt) / 1000;

    const row = document.createElement('div');
    row.className = 'metrics live';
    row.appendChild(span('breathe'));
    row.appendChild(metric('', money(t.chars), '字'));
    if (generation > 0.4) row.appendChild(metric('', fmt(1, t.chars / generation), '字/秒'));
    row.appendChild(metric('', fmt(1, total) + 's', ''));
    return row;
  }

  function paintLive(t) {
    if (!t) return;
    const now = performance.now();
    if (now - t.lastPaint < 90) return; // 别为了跳数字把主线程占满
    t.lastPaint = now;
    const row = liveRow(t);
    t.live.replaceWith(row);
    t.live = row; // 引用必须跟着换到新节点上，否则下一次 replaceWith 打在已摘除的旧节点上
  }

  /** 正文段落：第一段新建，之后往同一段里追加，直到来了一次工具调用把它「封口」。 */
  function appendText(t, delta) {
    if (!t) return;
    t.text += delta;
    t.chars += delta.length;
    if (!t.prose) {
      t.prose = document.createElement('div');
      t.prose.className = 'prose streaming';
      t.bot.insertBefore(t.prose, t.live);
    }
    // 流式期间只塞纯文本（textContent 不过 HTML 解析，最快也最安全），
    // 等这一段说完再整体换成渲染好的富文本 —— 见 sealProse()。
    t.prose.textContent = t.text;
    paintLive(t);
    scrollToEnd();
  }

  function sealProse(t) {
    if (!t || !t.prose) return;
    t.prose.classList.remove('streaming');
    t.prose.replaceChildren(renderRich(t.text));
    t.prose = null;
    t.text = '';
  }

  /** 一个工具调用块。流式期间和从库里恢复历史都用它，两边长得一样。 */
  function toolBox(step) {
    const box = document.createElement('details');
    box.className = 'tool';
    box.dataset.error = String(!!step.is_error);

    const summary = document.createElement('summary');
    summary.appendChild(span('tdot'));
    summary.appendChild(span('tname', step.tool));
    summary.appendChild(span('targs', formatArgs(step.arguments)));
    if (step.elapsed) summary.appendChild(span('ttime', fmt(2, step.elapsed) + 's'));
    box.appendChild(summary);

    const body = document.createElement('div');
    body.className = 'tbody';
    body.appendChild(span('label', step.is_error ? '错误' : '返回'));
    const pre = document.createElement('pre');
    pre.textContent = step.result;
    body.appendChild(pre);
    box.appendChild(body);
    return box;
  }

  function appendTool(t, step) {
    if (!t) return;
    sealProse(t); // 工具之后的文字属于新的一段，另起一块
    t.bot.insertBefore(toolBox(step), t.live);
    scrollToEnd();
  }

  /**
   * 一轮之后要追加的东西（记忆卡片、压缩提示）都走这里。
   *
   * 位置不一样：还在流式时新内容要插在「正在发生」那一行**前面**；
   * done 之后那一行已经换成了指标行，新内容该在它**下面**——答案、指标、
   * 然后才是「顺手记下了什么」。
   */
  function appendAfter(t, node) {
    if (!t) return;
    if (t.settled) t.bot.appendChild(node);
    else t.bot.insertBefore(node, t.live);
    scrollToEnd();
  }

  /**
   * 「已记住」卡片。自动抽取是回答完之后才跑的，所以卡片一定出现在指标行下面，
   * 而且带一个当场撤销的入口——用户看见它才知道刚才那句话被记下来了。
   */
  function renderMemo(t, facts) {
    if (!t || !facts.length) return;
    const box = document.createElement('div');
    box.className = 'memo';

    const head = document.createElement('div');
    head.className = 'memo-head';
    head.appendChild(icon('check'));
    const label = document.createElement('span');
    label.textContent = facts.length > 1 ? `记住了 ${facts.length} 条` : '记住了';
    head.appendChild(label);
    box.appendChild(head);

    const list = document.createElement('ul');
    list.className = 'memo-list';
    for (const fact of facts) {
      const item = document.createElement('li');
      const cat = document.createElement('span');
      cat.className = 'cat';
      cat.textContent = fact.category_label || '其他';
      const text = document.createElement('span');
      text.className = 'text';
      text.textContent = fact.content;
      item.append(cat, text);
      list.appendChild(item);
    }
    box.appendChild(list);

    const foot = document.createElement('div');
    foot.className = 'memo-foot';
    const undo = document.createElement('button');
    undo.type = 'button';
    undo.className = 'link-btn';
    undo.textContent = '撤销';
    undo.addEventListener('click', async () => {
      undo.disabled = true;
      let failed = 0;
      for (const fact of facts) {
        try {
          await postJSON('/api/memories/delete', { id: fact.id });
        } catch {
          failed++;
        }
      }
      box.remove();
      if (failed) toast(`有 ${failed} 条没撤销掉，可以去记忆页看看。`, { warn: true });
      else toast('已经忘掉了，之后的回答不会再带上它。');
    });
    const view = document.createElement('button');
    view.type = 'button';
    view.className = 'link-btn';
    view.textContent = '去记忆页看看';
    view.addEventListener('click', () => showView('memory'));
    foot.append(undo, view);
    box.appendChild(foot);

    appendAfter(t, box);
  }

  /** 历史被压成摘要了，说一声。用户有权知道早期对话不再逐字发给模型。 */
  function renderCompressed(t, data) {
    if (!t) return;
    const row = document.createElement('div');
    row.className = 'compressed';
    row.appendChild(icon('layers'));
    const text = document.createElement('span');
    text.textContent = `对话变长了，早期内容已经压成 ${data.chars} 字摘要（原文仍在本地，往上翻还看得见）。`;
    row.appendChild(text);
    appendAfter(t, row);
  }

  function formatArgs(args) {
    if (!args) return '';
    const parts = Object.entries(args).map(([key, value]) => {
      const text = typeof value === 'string' ? value : JSON.stringify(value);
      const clipped = text && text.length > 46 ? text.slice(0, 46) + '…' : text;
      return `${key}=${clipped}`;
    });
    return parts.join(' ');
  }

  function renderMetrics(t, metrics) {
    if (!t) return;
    sealProse(t);

    const row = document.createElement('div');
    row.className = 'metrics';
    const lines = [
      `${metrics.output_tokens} tokens ÷ 模型耗时 ${metrics.llm_seconds}s（来源：${metrics.window_source}，含首字等待）`,
      `整轮 ${metrics.elapsed}s · 首字 ${metrics.ttft}s · 解码窗口 ${metrics.decode_seconds}s`,
    ];
    if (metrics.decode_tok_per_s) {
      lines.push(`只看解码窗口约 ${fmt(1, metrics.decode_tok_per_s)} tok/s（厂商不是逐字吐时这个数会偏高）`);
    }
    lines.push(
      metrics.cache_read_input_tokens
        ? `命中提示词缓存 ${money(metrics.cache_read_input_tokens)} tokens`
        : '未命中提示词缓存'
    );
    row.title = lines.join('\n');

    const add = (...args) => row.appendChild(metric(...args));

    // 用户最关心的三个数排在最前：输入、输出、速度
    add('输入', money(metrics.input_tokens), 'tokens');
    add('输出', metrics.output_tokens ? money(metrics.output_tokens) : '—', 'tokens');
    if (metrics.tok_per_s) {
      add('速度', fmt(1, metrics.tok_per_s), 'tok/s', 'accent');
    } else if (metrics.chars_per_s) {
      // 厂商没回 token 统计时的兜底：如实标成字/秒，不伪装成 token 速度
      add('速度', '≈ ' + fmt(1, metrics.chars_per_s), '字/秒', 'accent');
    }
    if (metrics.cache_read_input_tokens) {
      add('缓存', money(metrics.cache_read_input_tokens), 'tokens');
    }
    add('首字', fmt(2, metrics.ttft), 's');
    add('', metrics.steps + ' 步', '');
    add('', fmt(2, metrics.elapsed), 's');
    if (metrics.stop_reason !== 'final') {
      add('', metrics.stop_reason_text, '', 'warn');
    }

    t.live.replaceWith(row);
    t.live = row;
    t.settled = true;
    t.bot.setAttribute('aria-busy', 'false');
    t.node.appendChild(span('stamp', stamp()));
    scrollToEnd();
  }

  function renderError(t, message) {
    if (!t) return;
    sealProse(t);
    const row = document.createElement('div');
    row.className = 'metrics';
    row.appendChild(metric('', message, '', 'warn'));
    t.live.replaceWith(row);
    t.live = row;
    t.settled = true;
    t.bot.setAttribute('aria-busy', 'false');
    scrollToEnd();
  }

  // ==================================================================== 交互

  function setRunning(running) {
    state.running = running;
    el.send.hidden = running;
    el.stop.hidden = !running;
    el.input.disabled = running;
    el.chip.classList.toggle('live', running);
    if (!running) el.input.focus();
  }

  //: 每一轮一个号。done 之后输入框就解锁了，用户可能立刻开始下一轮，
  //: 而上一轮的连接还要过一会儿才关 —— 收尾时靠它判断「我还是最新那一轮吗」，
  //: 否则上一轮结束会把正在跑的这一轮标成「没在跑」。
  let runSeq = 0;

  async function ask(question) {
    if (state.running || !question.trim()) return;

    const run = ++runSeq;
    const turn = makeTurn(question);
    setRunning(true);
    el.input.value = '';
    autosize();

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        // 带上会话 id：页面上开着哪个会话，这一问就写进哪个。
        body: JSON.stringify({ message: question, session: state.sessionId }),
      });
      if (bouncedToLogin(response)) return;
      if (!response.ok || !response.body) {
        renderError(turn, `请求失败（HTTP ${response.status}）`);
        return;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let cut;
        while ((cut = buffer.indexOf('\n\n')) >= 0) {
          handleFrame(buffer.slice(0, cut), turn);
          buffer = buffer.slice(cut + 2);
        }
      }
      sealProse(turn);
    } catch (error) {
      renderError(turn, `连接中断：${error.message}`);
    } finally {
      sealProse(turn);
      if (run === runSeq) setRunning(false);
    }
  }

  function handleFrame(frame, turn) {
    let name = 'message';
    const dataLines = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event: ')) name = line.slice(7).trim();
      else if (line.startsWith('data: ')) dataLines.push(line.slice(6));
    }
    if (!dataLines.length) return;

    let data;
    try {
      data = JSON.parse(dataLines.join('\n'));
    } catch {
      return;
    }

    if (name === 'text') appendText(turn, data.delta || '');
    else if (name === 'step') appendTool(turn, data);
    else if (name === 'done') {
      renderMetrics(turn, data.metrics);
      // done 是「答案已经给完了」的信号。后面那几件事（抽记忆、压历史、
      // 算上下文）是锦上添花，不该让用户对着一个锁住的输入框等它们。
      if (data.session) state.sessionId = data.session.id;
      setRunning(false);
      refreshSessions();
      loadContext();
    } else if (name === 'memory') renderMemo(turn, data.facts || []);
    else if (name === 'compressed') renderCompressed(turn, data);
    else if (name === 'context') setContext(data);
    else if (name === 'error') renderError(turn, data.message || '出错了');
  }

  async function cancel() {
    try {
      await fetch('/api/cancel', { method: 'POST' });
    } catch {
      /* 连接都没了，这一轮本来也停了 */
    }
  }

  /** 输入框跟着内容长高，最多 200px（再多就该自己滚了）。 */
  function autosize() {
    el.input.style.height = 'auto';
    el.input.style.height = Math.min(el.input.scrollHeight, 200) + 'px';
  }

  /** 清空**当前会话**的对话历史。会话本身、以及长期记忆都留着。 */
  async function reset() {
    if (state.running) await cancel();
    try {
      const result = await postJSON('/api/reset', {});
      renderTranscript(result.session);
      await refreshSessions();
      toast('这个会话的历史清空了，长期记忆还在。');
    } catch (error) {
      toast(`清空失败：${error.message}`, { warn: true });
    }
  }

  // ==================================================================== 图标
  //
  // 设计要求点名了 Lucide / Phosphor 那一路（细线、圆角、stroke-width 1.5、
  // 颜色跟随文字），这里没有引它们：整个前端是零依赖的，为几个图标挂一个 CDN
  // 脚本，等于把「CDN 挂了界面照常能用」这条性质搭进去。所以路径数据直接写在
  // 这里，形状照着那种风格画（stroke-width 和尺寸在 style.css 的 .ico 里统一给）。

  const SVG_NS = 'http://www.w3.org/2000/svg';
  const ICONS = {
    plus: 'M12 5v14M5 12h14',
    close: 'M6 6l12 12M18 6L6 18',
    check: 'M5 12.5l4.5 4.5L19 7',
    info: 'M12 4.5a7.5 7.5 0 1 0 0 15 7.5 7.5 0 0 0 0-15zM12 11v5M12 8h.01',
    alert: 'M12 4l8.5 15h-17L12 4zM12 10v4M12 16.5h.01',
    trash: 'M4 7h16M9.5 7V4.5h5V7M6.5 7l1 12.5h9l1-12.5M10 10.5v6M14 10.5v6',
    pencil: 'M4 20h4L19.5 8.5a2.1 2.1 0 0 0-3-3L5 17v3zM14.5 6.5l3 3',
    top: 'M12 19V6M7 11l5-5 5 5M5 20h14',
    book: 'M4 5.5A2.5 2.5 0 0 1 6.5 3H19v18H6.5A2.5 2.5 0 0 1 4 18.5v-13zM9 3v18',
    message: 'M4 5h16v11H9.5L4 20V5z',
    sliders:
      'M4 8h8M16 8h4M4 16h4M12 16h8' +
      'M12 8a2 2 0 1 0 4 0 2 2 0 1 0-4 0' +
      'M8 16a2 2 0 1 0 4 0 2 2 0 1 0-4 0',
    slash: 'M12 4.5a7.5 7.5 0 1 0 0 15 7.5 7.5 0 0 0 0-15zM6.7 6.7l10.6 10.6',
    panel: 'M4 5h16v14H4zM9.5 5v14',
    layers: 'M4 7h16M4 12h16M4 17h10',
  };

  function icon(name, cls) {
    const box = document.createElement('span');
    box.className = 'ico' + (cls ? ' ' + cls : '');
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('aria-hidden', 'true');
    const path = document.createElementNS(SVG_NS, 'path');
    path.setAttribute('d', ICONS[name] || ICONS.info);
    svg.appendChild(path);
    box.appendChild(svg);
    return box;
  }

  /** 把 HTML 里的 `<span data-icon="x">` 占位换成真的图标。 */
  function fillIcons(root) {
    for (const holder of root.querySelectorAll('[data-icon]')) {
      holder.replaceWith(icon(holder.dataset.icon));
    }
  }

  /** 一个只有图标的按钮。侧栏条目和记忆条目上那些「悬停才出现」的操作都用它。 */
  function iconButton(name, label, onClick, cls) {
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'icon-btn' + (cls ? ' ' + cls : '');
    button.title = label;
    button.setAttribute('aria-label', label);
    button.appendChild(icon(name));
    button.addEventListener('click', (event) => {
      event.stopPropagation();
      onClick();
    });
    return button;
  }

  // ==================================================================== 接口

  /**
   * 401 = 没登录（或者 30 天到期了）。这时候该做的是把人送回登录页，
   * 而不是让他对着一个「未连接」的界面猜哪里坏了。
   *
   * 返回 true 表示已经跳走了，调用方直接返回、别再往下渲染。
   * 换地址用 replace 而不是 assign：登录页不该占一格后退历史。
   */
  function bouncedToLogin(response) {
    if (response.status !== 401) return false;
    location.replace('/login');
    return true;
  }

  async function getJSON(path) {
    const response = await fetch(path);
    if (bouncedToLogin(response)) throw new Error('需要登录');
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    return response.json();
  }

  /**
   * POST 一个会改状态的接口。失败时把服务端那句话抛出来。
   *
   * 服务端把「用户填错了」翻成 400 并带一句人话（「标题不能为空」），这里要是
   * 一律报「请求失败」，那句人话就白写了。500 不带人话，退回状态码就行 ——
   * 那种情况本来也不该指望界面能解释清楚。
   */
  async function postJSON(path, body) {
    const response = await fetch(path, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
    });
    let data = null;
    try {
      data = await response.json();
    } catch {
      /* 没回 JSON（连接断了、被代理截了），下面按状态码处理 */
    }
    if (bouncedToLogin(response)) throw new Error('需要登录');
    if (!response.ok) throw new Error((data && data.error) || `HTTP ${response.status}`);
    return data || {};
  }

  // ==================================================================== 提示条
  //
  // 三种轻量反馈，各管一件事，不混用：
  //   toast   —— 刚才那件事做成了（右下角，几秒后自己消失）
  //   notice  —— 界面上发生了什么变化（顶部一条，说明「你现在看到的是哪个会话」）
  //   dialog  —— 不可撤销、需要停一下的操作（清空全部、忘掉一条记忆）

  function toast(text, options) {
    const opts = options || {};
    const box = document.createElement('div');
    box.className = 'toast';
    box.appendChild(icon(opts.warn ? 'alert' : 'check', opts.warn ? 'warn' : 'ok'));
    const label = document.createElement('span');
    label.textContent = text;
    box.appendChild(label);

    let gone = false;
    const dismiss = () => {
      if (gone) return;
      gone = true;
      clearTimeout(timer);
      box.classList.add('leaving');
      setTimeout(() => box.remove(), 200);
    };

    if (opts.action) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'link-btn';
      button.textContent = opts.action.label;
      button.addEventListener('click', () => {
        dismiss();
        opts.action.run();
      });
      box.appendChild(button);
    }

    el.toastHost.appendChild(box);
    const timer = setTimeout(dismiss, opts.action ? 9000 : 3200);
    return dismiss;
  }

  /** 顶部那条「已切换到……」。几秒后自己收回去，也可以手动关。 */
  function noticeSession(title) {
    el.noticeText.replaceChildren();
    el.noticeText.appendChild(document.createTextNode('已切换到 '));
    const name = document.createElement('b');
    name.textContent = title;
    el.noticeText.appendChild(name);
    el.noticeText.appendChild(document.createTextNode('，下面是它的完整历史。'));
    el.notice.hidden = false;
    clearTimeout(state.noticeTimer);
    state.noticeTimer = setTimeout(() => {
      el.notice.hidden = true;
    }, 4200);
  }

  /**
   * 需要停下来想一想的操作走这里。确认 resolve(true)，取消 / Esc / 点背景
   * resolve(false)。
   *
   * 给了 ``confirmWord`` 就要求一字不差地打出来才让点确认 —— 这是给「清空全部
   * 会话」那种不可撤销的操作准备的。界面上挡一次手滑，服务端还会再挡一次
   * （见 clear_all）：前端校验挡不住绕过界面直接发请求的人。
   */
  function confirmDialog(options) {
    return new Promise((resolve) => {
      const panel = el.dialogPanel;
      panel.replaceChildren();

      const title = document.createElement('h3');
      title.textContent = options.title;
      panel.appendChild(title);

      if (options.body) {
        const body = document.createElement('p');
        body.textContent = options.body;
        panel.appendChild(body);
      }

      let input = null;
      if (options.confirmWord) {
        input = document.createElement('input');
        input.type = 'text';
        input.autocomplete = 'off';
        input.spellcheck = false;
        input.placeholder = options.confirmWord;
        panel.appendChild(input);
        const note = document.createElement('p');
        note.className = 'dialog-note';
        note.textContent = `请输入「${options.confirmWord}」以确认`;
        panel.appendChild(note);
      }

      const buttons = document.createElement('div');
      buttons.className = 'dialog-btns';
      const cancelBtn = document.createElement('button');
      cancelBtn.type = 'button';
      cancelBtn.textContent = '取消';
      const goBtn = document.createElement('button');
      goBtn.type = 'button';
      goBtn.className = 'go';
      goBtn.textContent = options.confirmText || '确定';
      buttons.append(cancelBtn, goBtn);
      panel.appendChild(buttons);

      const close = (answer) => {
        el.dialog.hidden = true;
        document.removeEventListener('keydown', onKey);
        el.dialog.removeEventListener('click', onBackdrop);
        resolve(answer);
      };
      const onKey = (event) => {
        if (event.key === 'Escape') close(false);
        else if (event.key === 'Enter' && !goBtn.disabled) close(true);
      };
      const onBackdrop = (event) => {
        if (event.target === el.dialog) close(false);
      };
      const sync = () => {
        goBtn.disabled = !!options.confirmWord && input.value.trim() !== options.confirmWord;
      };

      if (input) {
        input.addEventListener('input', sync);
        sync();
      }
      cancelBtn.addEventListener('click', () => close(false));
      goBtn.addEventListener('click', () => {
        if (!goBtn.disabled) close(true);
      });

      el.dialog.hidden = false;
      document.addEventListener('keydown', onKey);
      el.dialog.addEventListener('click', onBackdrop);
      // 焦点落在取消上，不是确认上：回车连打两下不该把东西删掉
      (input || cancelBtn).focus();
    });
  }

  // ==================================================================== 视图切换

  function showView(name) {
    state.view = name;
    for (const tab of el.sideTabs) tab.classList.toggle('current', tab.dataset.view === name);
    el.stream.hidden = name !== 'chat';
    el.pageMemory.hidden = name !== 'memory';
    el.pageSettings.hidden = name !== 'settings';
    el.app.classList.remove('side-open');
    el.scrim.hidden = true;

    if (name === 'chat') {
      // 回到对话页时重放一次上下文状态：设置项里那个「显示上下文状态」是在
      // **另一个页面**上关掉的，不在这里补一次，就得等下一轮回答才生效——
      // 而设置页自己写着「改完立刻生效」。
      syncContext();
      scrollToEnd(true);
      el.input.focus();
    } else if (name === 'memory') {
      loadMemories();
    } else {
      loadSettings();
    }
  }

  function pageHead(title, sub) {
    const box = document.createElement('div');
    box.className = 'page-head';
    const head = document.createElement('h2');
    head.textContent = title;
    const text = document.createElement('p');
    text.textContent = sub;
    box.append(head, text);
    return box;
  }

  // ==================================================================== 会话侧栏

  async function refreshSessions() {
    let listing;
    try {
      listing = await getJSON('/api/sessions');
    } catch {
      return; // 侧栏拉不到就维持现状：一块导航不该为网络问题弹错误
    }
    state.sessions = listing;
    renderSessions(listing);
  }

  function renderSessions(listing) {
    state.sessions = listing;
    el.sideList.replaceChildren();
    if (!listing.groups.length) {
      const empty = document.createElement('div');
      empty.className = 'side-empty';
      empty.textContent = '还没有会话。';
      el.sideList.appendChild(empty);
      return;
    }
    for (const group of listing.groups) {
      const head = document.createElement('div');
      head.className = 'side-group';
      head.textContent = group.label;
      el.sideList.appendChild(head);
      for (const item of group.items) el.sideList.appendChild(sessionRow(item));
    }
  }

  function sessionRow(item) {
    const wrap = document.createElement('div');
    wrap.className =
      'side-item' + (item.current ? ' current' : '') + (item.pinned ? ' pinned' : '');

    const row = document.createElement('div');
    row.className = 'side-row';

    const open = document.createElement('button');
    open.type = 'button';
    open.className = 'side-open';
    const title = document.createElement('span');
    title.className = 'side-title';
    title.textContent = item.title;
    title.title = item.title;
    const meta = document.createElement('span');
    meta.className = 'side-meta';
    meta.textContent = item.turns ? `${item.turns} 轮 · ${item.age}` : item.age;
    open.append(title, meta);
    open.addEventListener('click', () => openSession(item.id));
    if (item.current) open.setAttribute('aria-current', 'true');

    const acts = document.createElement('div');
    acts.className = 'side-acts';
    acts.appendChild(
      iconButton(
        item.pinned ? 'top' : 'top',
        item.pinned ? '取消置顶' : '置顶',
        () => pinSession(item, !item.pinned),
        item.pinned ? 'side-pin' : ''
      )
    );
    acts.appendChild(iconButton('pencil', '重命名', () => startRename(wrap, item)));
    acts.appendChild(iconButton('trash', '删除会话', () => startDelete(wrap, item)));

    row.append(open, acts);
    wrap.appendChild(row);
    return wrap;
  }

  /** 点侧栏里的一条：把那个会话整段铺到主区。 */
  async function openSession(sessionId) {
    if (sessionId === state.sessionId) {
      el.app.classList.remove('side-open');
      return;
    }
    // 换会话前先把在跑的那一轮停掉：答案会落进旧会话，而界面已经翻页了
    if (state.running) await cancel();

    let detail;
    try {
      detail = await postJSON('/api/sessions/select', { id: sessionId });
    } catch (error) {
      toast(`切换会话失败：${error.message}`, { warn: true });
      return;
    }
    state.sessionId = detail.id;
    renderTranscript(detail);
    await refreshSessions();
    noticeSession(detail.title);
    el.app.classList.remove('side-open');
    el.scrim.hidden = true;
    el.input.focus();
  }

  /** 把落盘的历史铺回消息流。切会话时**整段换掉**——留着上一场的回答是最糟的错觉。 */
  function renderTranscript(detail) {
    el.turns.replaceChildren();
    const messages = detail.messages || [];
    el.empty.hidden = messages.length > 0;

    let bot = null;
    for (const item of messages) {
      if (item.role === 'user') {
        const turn = document.createElement('article');
        turn.className = 'turn';
        const row = document.createElement('div');
        row.className = 'msg-user';
        const bubble = document.createElement('div');
        bubble.className = 'bubble';
        bubble.textContent = item.text;
        row.appendChild(bubble);
        turn.appendChild(row);
        bot = document.createElement('div');
        bot.className = 'msg-bot';
        turn.appendChild(bot);
        el.turns.appendChild(turn);
      } else if (!bot) {
        continue; // 历史里第一组不是用户提问（旧数据），跳过而不是造一个没有提问的回答
      } else if (item.role === 'assistant') {
        const prose = document.createElement('div');
        prose.className = 'prose';
        prose.replaceChildren(renderRich(item.text));
        bot.appendChild(prose);
      } else if (item.role === 'tool') {
        bot.appendChild(toolBox(item));
      }
    }

    el.crumb.textContent = detail.title;
    setContext(detail.context);
    scrollToEnd(true);
  }

  async function newSession() {
    if (state.running) await cancel();
    let detail;
    try {
      detail = await postJSON('/api/sessions/new', {});
    } catch (error) {
      toast(`新建会话失败：${error.message}`, { warn: true });
      return;
    }
    state.sessionId = detail.id;
    renderTranscript(detail);
    await refreshSessions();
    showView('chat');
    toast('已新建会话，这是一个干净的窗口。');
    el.input.focus();
  }

  async function pinSession(item, pinned) {
    try {
      renderSessions(await postJSON('/api/sessions/pin', { id: item.id, pinned }));
    } catch (error) {
      toast(`置顶失败：${error.message}`, { warn: true });
    }
  }

  function startRename(wrap, item) {
    const row = wrap.querySelector('.side-row');
    const input = document.createElement('input');
    input.className = 'side-rename';
    input.value = item.title;
    input.maxLength = 60;
    row.replaceWith(input);
    input.focus();
    input.select();

    let done = false;
    const finish = async (save) => {
      if (done) return;
      done = true;
      const title = input.value.trim();
      if (!save || !title || title === item.title) {
        await refreshSessions();
        return;
      }
      try {
        renderSessions(await postJSON('/api/sessions/rename', { id: item.id, title }));
        if (item.current) el.crumb.textContent = title;
      } catch (error) {
        toast(`重命名失败：${error.message}`, { warn: true });
        await refreshSessions();
      }
    };

    input.addEventListener('keydown', (event) => {
      // stopPropagation：不然 Esc 会一路冒到 document 上，顺手把在跑的那一轮停掉
      if (event.key === 'Enter') {
        event.preventDefault();
        event.stopPropagation();
        finish(true);
      } else if (event.key === 'Escape') {
        event.preventDefault();
        event.stopPropagation();
        finish(false);
      }
    });
    input.addEventListener('blur', () => finish(true));
  }

  function startDelete(wrap, item) {
    const box = document.createElement('div');
    box.className = 'side-confirm';

    const ask = document.createElement('p');
    ask.textContent = `删除「${item.title}」？这个会话的对话历史会一起删掉，删了找不回来。`;
    box.appendChild(ask);

    const buttons = document.createElement('div');
    buttons.className = 'side-confirm-btns';
    const cancel = document.createElement('button');
    cancel.type = 'button';
    cancel.textContent = '取消';
    const go = document.createElement('button');
    go.type = 'button';
    go.className = 'go';
    go.textContent = '删除';
    buttons.append(cancel, go);
    box.appendChild(buttons);
    wrap.appendChild(box);
    cancel.focus();

    // 「同时删除这个会话产生的记忆」这个勾只在真有记忆时才出现：
    // 一个永远勾不动、或者勾了也没东西可删的开关，比没有更让人困惑。
    let removeMemories = false;
    getJSON('/api/memories')
      .then((memories) => {
        const mine = memories.categories
          .flatMap((group) => group.items)
          .filter((fact) => fact.source_session === item.id);
        if (!mine.length) return;
        const label = document.createElement('label');
        label.className = 'side-check';
        const check = document.createElement('input');
        check.type = 'checkbox';
        check.checked = true;
        removeMemories = true;
        check.addEventListener('change', () => {
          removeMemories = check.checked;
        });
        const text = document.createElement('span');
        text.textContent = `同时删除这个会话产生的记忆（${mine.length} 条）`;
        label.append(check, text);
        box.insertBefore(label, buttons);
      })
      .catch(() => {});

    cancel.addEventListener('click', () => {
      box.remove();
      wrap.querySelector('.side-open')?.focus();
    });
    go.addEventListener('click', async () => {
      go.disabled = true;
      try {
        const wasCurrent = item.current;
        const result = await postJSON('/api/sessions/delete', {
          id: item.id,
          with_memories: removeMemories,
        });
        renderSessions(result.sessions);
        const dropped = result.memories_dropped || 0;
        toast(
          dropped
            ? `会话已删除，连带忘掉了 ${dropped} 条记忆。`
            : '会话已删除，长期记忆没动。'
        );
        if (wasCurrent && result.current) {
          const detail = await postJSON('/api/sessions/select', { id: result.current });
          state.sessionId = detail.id;
          renderTranscript(detail);
        }
      } catch (error) {
        toast(`删除失败：${error.message}`, { warn: true });
        go.disabled = false;
      }
    });
  }

  async function clearAllSessions() {
    const yes = await confirmDialog({
      title: '清空全部会话？',
      body: `所有对话历史都会删掉（共 ${state.sessions.count} 个会话），删了找不回来。长期记忆不受影响——那是「你是谁」，不是「你聊过什么」。`,
      confirmWord: '确认删除',
      confirmText: '清空',
    });
    if (!yes) return;

    try {
      const result = await postJSON('/api/sessions/clear', { confirm: '确认删除' });
      renderSessions(result.sessions);
      if (state.running) await cancel();
      const detail = await getJSON('/api/session');
      state.sessionId = detail.id;
      renderTranscript(detail);
      showView('chat');
      toast(`已清空 ${result.cleared} 个会话，长期记忆都留着。`);
    } catch (error) {
      toast(`清空失败：${error.message}`, { warn: true });
    }
  }

  // ==================================================================== 记忆页

  async function loadMemories() {
    let data;
    try {
      data = await getJSON('/api/memories');
    } catch (error) {
      toast(`记忆没读出来：${error.message}`, { warn: true });
      return;
    }
    renderMemories(data);
  }

  function renderMemories(data) {
    el.pageMemory.replaceChildren();
    el.pageMemory.appendChild(
      pageHead(
        '记忆',
        `长期记忆是唯一跨会话的东西：新建会话时会带上它们（除非关掉了继承）。` +
          `目前 ${data.count} 条，其中 ${data.enabled} 条生效中。`
      )
    );

    const add = document.createElement('div');
    add.className = 'mem-add';
    const input = document.createElement('input');
    input.type = 'text';
    input.maxLength = 200;
    input.placeholder = '手动记一条，比如「我在用 Windows 开发」';
    const select = document.createElement('select');
    for (const group of data.categories) {
      const option = document.createElement('option');
      option.value = group.key;
      option.textContent = group.label;
      select.appendChild(option);
    }
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = '记住';
    const submit = async () => {
      const content = input.value.trim();
      if (!content) return;
      button.disabled = true;
      try {
        // 写接口回的是 {fact, memories}：那一份 fact 给 toast 用得上，
        // 但整页重画要的是 memories 本身。
        renderMemories((await postJSON('/api/memories/add', { content, category: select.value })).memories);
        toast('记住了。');
      } catch (error) {
        toast(`没记住：${error.message}`, { warn: true });
        button.disabled = false;
      }
    };
    button.addEventListener('click', submit);
    input.addEventListener('keydown', (event) => {
      if (event.key === 'Enter') {
        event.preventDefault();
        submit();
      }
    });
    add.append(input, select, button);
    el.pageMemory.appendChild(add);

    let any = false;
    for (const group of data.categories) {
      if (!group.items.length) continue;
      any = true;
      const section = document.createElement('section');
      section.className = 'mem-cat';
      const head = document.createElement('div');
      head.className = 'mem-cat-head';
      const label = document.createElement('span');
      label.className = 'mem-cat-label';
      label.textContent = group.label;
      const count = document.createElement('span');
      count.className = 'mem-cat-count';
      count.textContent = String(group.items.length);
      head.append(label, count);
      section.appendChild(head);
      for (const item of group.items) section.appendChild(memItem(item));
      el.pageMemory.appendChild(section);
    }

    if (!any) {
      const empty = document.createElement('p');
      empty.className = 'mem-empty';
      empty.textContent =
        '还没有记住任何东西。聊到你的偏好、正在做的事时，我会自己记下来并给你看一眼。';
      el.pageMemory.appendChild(empty);
    }
  }

  function memItem(item) {
    const row = document.createElement('div');
    row.className = 'mem-item' + (item.disabled ? ' disabled' : '');

    const body = document.createElement('div');
    body.className = 'mem-body';
    const content = document.createElement('div');
    content.className = 'mem-content';
    content.textContent = item.content;
    body.appendChild(content);

    // 停用不是删除：它还在列表里，只是不再注入。所以要有一个看得见的标记，
    // 而不是整条消失——用户会以为被删了。
    if (item.disabled) {
      const off = document.createElement('span');
      off.className = 'mem-off';
      off.textContent = '已停用';
      body.appendChild(off);
    }

    const acts = document.createElement('div');
    acts.className = 'mem-acts';
    acts.appendChild(iconButton('pencil', '编辑', () => startEdit(row, item)));
    acts.appendChild(
      iconButton(
        item.disabled ? 'check' : 'slash',
        item.disabled ? '恢复' : '停用',
        () => patchFact(item, { disabled: !item.disabled })
      )
    );
    acts.appendChild(iconButton('trash', '忘掉这条', () => forgetFact(item)));
    body.appendChild(acts);
    row.appendChild(body);

    const meta = document.createElement('div');
    meta.className = 'mem-meta';
    const origin = document.createElement('span');
    origin.textContent = `来自：${item.origin}`;
    const age = document.createElement('span');
    age.textContent = `更新于 ${item.age}`;
    const version = document.createElement('span');
    version.textContent = `第 ${item.version} 版`;
    meta.append(origin, age, version);
    row.appendChild(meta);
    return row;
  }

  async function patchFact(item, changes) {
    try {
      renderMemories((await postJSON('/api/memories/update', { id: item.id, ...changes })).memories);
      if (changes.disabled === true) toast('已停用，下一轮开始不再带上它。');
      else if (changes.disabled === false) toast('已恢复。');
    } catch (error) {
      toast(`没改成：${error.message}`, { warn: true });
    }
  }

  function startEdit(row, item) {
    const panel = document.createElement('div');
    panel.className = 'mem-edit';
    const textarea = document.createElement('textarea');
    textarea.value = item.content;
    textarea.maxLength = 200;
    panel.appendChild(textarea);

    const controls = document.createElement('div');
    controls.className = 'mem-edit-row';
    const select = document.createElement('select');
    for (const [key, label] of [
      ['identity', '身份'],
      ['preference', '偏好'],
      ['project', '项目'],
      ['other', '其他'],
    ]) {
      const option = document.createElement('option');
      option.value = key;
      option.textContent = label;
      if (key === item.category) option.selected = true;
      select.appendChild(option);
    }
    const spacer = document.createElement('span');
    spacer.className = 'spacer';
    const cancel = document.createElement('button');
    cancel.type = 'button';
    cancel.textContent = '取消';
    const save = document.createElement('button');
    save.type = 'button';
    save.className = 'save';
    save.textContent = '保存';
    controls.append(select, spacer, cancel, save);
    panel.appendChild(controls);
    row.appendChild(panel);
    textarea.focus();
    textarea.setSelectionRange(textarea.value.length, textarea.value.length);

    cancel.addEventListener('click', () => panel.remove());
    textarea.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') {
        event.stopPropagation();
        panel.remove();
      } else if (event.key === 'Enter' && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        save.click();
      }
    });
    save.addEventListener('click', async () => {
      const content = textarea.value.trim();
      if (!content) {
        toast('内容不能是空的。', { warn: true });
        return;
      }
      await patchFact(item, { content, category: select.value });
      toast('改好了，下一轮开始用新的。');
    });
  }

  async function forgetFact(item) {
    const yes = await confirmDialog({
      title: '忘掉这条记忆？',
      body: `「${item.content}」`,
      confirmText: '忘掉',
    });
    if (!yes) return;
    try {
      renderMemories((await postJSON('/api/memories/delete', { id: item.id })).memories);
      toast('已经忘掉了。');
    } catch (error) {
      toast(`没删掉：${error.message}`, { warn: true });
    }
  }

  // ==================================================================== 设置页

  async function loadSettings() {
    let data;
    try {
      data = await getJSON('/api/settings');
    } catch (error) {
      toast(`设置没读出来：${error.message}`, { warn: true });
      return;
    }
    renderSettings(data);
  }

  function renderSettings(data) {
    // 顺手把开机时抓的那份 /api/info.settings 刷新掉。setContext 是拿它判断
    // 「显示上下文状态」开没开的——那份是启动时抓的，改完设置不同步，
    // 开关就得等下一次回答才生效。
    state.info.settings = Object.fromEntries(data.items.map((item) => [item.key, item.value]));

    el.pageSettings.replaceChildren();
    el.pageSettings.appendChild(
      pageHead('设置', '这些开关存在本地数据库里，重启之后还在。改完立刻生效，不用重启。')
    );

    for (const item of data.items) {
      const row = document.createElement('div');
      row.className = 'set-item';
      const text = document.createElement('div');
      text.className = 'set-text';
      const label = document.createElement('div');
      label.className = 'set-label';
      label.textContent = item.label;
      const hint = document.createElement('div');
      hint.className = 'set-hint';
      hint.textContent = item.hint;
      text.append(label, hint);

      const toggle = document.createElement('button');
      toggle.type = 'button';
      toggle.className = 'switch';
      toggle.setAttribute('role', 'switch');
      toggle.setAttribute('aria-checked', String(item.value));
      toggle.setAttribute('aria-label', item.label);
      toggle.addEventListener('click', async () => {
        const next = toggle.getAttribute('aria-checked') !== 'true';
        toggle.setAttribute('aria-checked', String(next)); // 先动，失败再翻回来
        try {
          const fresh = await postJSON('/api/settings', { values: { [item.key]: next } });
          renderSettings(fresh); // 顺带刷新 state.info.settings
          syncContext(); // 立刻按新设置重画上下文状态行
        } catch (error) {
          toggle.setAttribute('aria-checked', String(!next));
          toast(`没改成功：${error.message}`, { warn: true });
        }
      });

      row.append(text, toggle);
      el.pageSettings.appendChild(row);
    }

    const note = document.createElement('p');
    note.className = 'page-note';
    note.textContent =
      '长期记忆和会话历史都存在本机的 SQLite 里，不上传任何地方。' +
      '关掉「自动保存会话」只影响新写入的历史，已经存下的不会因此被删。';
    el.pageSettings.appendChild(note);

    // 只有配了口令才有「退出」这回事。没配就别摆一个点了没反应的按钮——
    // 本机用法下重新输一遍口令纯属自找麻烦。
    if (state.info.auth) {
      const row = document.createElement('div');
      row.className = 'set-item';
      const text = document.createElement('div');
      text.className = 'set-text';
      const label = document.createElement('div');
      label.className = 'set-label';
      label.textContent = '退出登录';
      const hint = document.createElement('div');
      hint.className = 'set-hint';
      hint.textContent = '清掉这台浏览器上的登录状态，下次打开要重新输口令。';
      text.append(label, hint);

      const out = document.createElement('button');
      out.type = 'button';
      out.className = 'link-btn danger';
      out.textContent = '退出';
      out.addEventListener('click', async () => {
        out.disabled = true;
        try {
          await postJSON('/api/logout', {});
        } catch {
          // 服务端没答上也照样回登录页：cookie 清没清掉是服务端的事，
          // 但这一页不该继续开着。
        }
        location.replace('/login');
      });

      row.append(text, out);
      el.pageSettings.appendChild(row);
    }
  }

  // ==================================================================== 上下文状态

  /** 拉一次当前会话的上下文状态。切会话、跑完一轮、改设置之后都该刷。 */
  async function loadContext() {
    try {
      setContext(await getJSON('/api/context'));
    } catch {
      /* 状态行是装饰，拉不到就不显示 */
    }
  }

  function syncContext() {
    setContext(state.context);
  }

  /**
   * 输入框上方那行「上下文约 N tokens」，以及展开后的分层明细。
   *
   * 比例条用的是**字符**预算（window_chars / budget_chars），不是 token：
   * 真正决定历史被裁到哪里的就是那个字符上限，拿 tokens 去除以字符预算会得到
   * 一个凭空造出来的百分比。token 数只出现在文字和明细里，那里本来就是估算值。
   */
  function setContext(data) {
    state.context = data;
    const show = data && state.info.settings && state.info.settings.show_context;
    if (!show) {
      el.ctx.hidden = true;
      return;
    }
    el.ctx.hidden = false;

    const tokens = data.degraded ? null : data.tokens;
    el.ctxText.replaceChildren();
    el.ctxText.appendChild(document.createTextNode('上下文 '));
    const strong = document.createElement('b');
    strong.textContent = tokens === null ? `${data.messages} 条历史` : money(tokens);
    el.ctxText.appendChild(strong);
    el.ctxText.appendChild(document.createTextNode(tokens === null ? '' : ' tokens'));

    const total = data.budget_chars || 0;
    const used = data.window_chars || 0;
    const ratio = total > 0 ? Math.min(used / total, 1) : 0;
    el.ctxBar.replaceChildren();
    const fill = document.createElement('span');
    fill.style.width = (ratio * 100).toFixed(1) + '%';
    el.ctxBar.appendChild(fill);
    el.ctxBar.classList.toggle('warn', ratio > 0.8);
    el.ctxLine.title = total
      ? `窗口 ${money(used)} / ${money(total)} 字符，指上去看各层明细`
      : '指上去看各层明细';

    el.ctxDetail.replaceChildren();
    if (data.degraded) {
      const row = document.createElement('div');
      row.className = 'ctx-foot';
      row.textContent =
        '这个会话还没在本进程里跑过，分层统计要等下一次提问才算得出来；上面那条消息数是留在库里的历史。';
      el.ctxDetail.appendChild(row);
      return;
    }

    const max = Math.max(1, ...data.layers.map((layer) => layer.tokens));
    for (const layer of data.layers) {
      const row = document.createElement('div');
      row.className = 'ctx-row';
      row.dataset.layer = layer.key;
      const name = document.createElement('span');
      name.className = 'name';
      name.textContent = layer.label;
      const track = document.createElement('span');
      track.className = 'track';
      const fillBar = document.createElement('span');
      fillBar.className = 'fill';
      fillBar.style.width = ((layer.tokens / max) * 100).toFixed(1) + '%';
      track.appendChild(fillBar);
      const num = document.createElement('span');
      num.className = 'num';
      num.textContent = money(layer.tokens);
      row.append(name, track, num);
      el.ctxDetail.appendChild(row);
    }

    const foot = document.createElement('div');
    foot.className = 'ctx-foot';
    const lines = [
      `窗口 ${money(used)} / ${money(total)} 字符 · 保留 ${data.window_messages} 条消息 · 已丢弃 ${data.dropped_messages} 条`,
    ];
    if (data.summary) {
      lines.push(`历史摘要已覆盖到第 ${data.summary_upto} 条（不再进窗口）`);
    }
    if (data.pending_tokens) {
      lines.push(`还有约 ${money(data.pending_tokens)} tokens 的旧历史没压进摘要，下次回答完会滚一次`);
    }
    if (!data.auto_summarize) {
      lines.push('「自动压缩历史」关着：超出窗口的对话会直接丢弃，不压成摘要');
    }
    if (!data.memory_enabled) {
      lines.push('「长期记忆」关着：上面那层长期记忆没有注入');
    }
    foot.textContent = lines.join('；');
    el.ctxDetail.appendChild(foot);
  }

  // ==================================================================== 启动

  /** 一条建议按钮。`label` 是按钮上显示的，`payload` 是点下去真正发出去的。 */
  function suggestionButton(label, payload) {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.addEventListener('click', () => {
      // 输入框里填的是**真正发出去的那一份**，不是按钮上那行简写：
      // 热点推送会缀上来源链接，得让人看见到底发了什么。
      el.input.value = payload;
      autosize();
      ask(payload);
    });
    return button;
  }

  /** 静态建议：`/api/info` 给的一串问句（服务端已按「工具注册了才出现」筛过）。 */
  function renderSuggestions() {
    el.suggestions.replaceChildren();
    for (const question of state.info.suggestions) {
      el.suggestions.appendChild(suggestionButton(question, question));
    }
  }

  /** 实时热点。拉到了就按分组铺开，拉不到就什么都不动，留着静态建议。 */
  async function renderHot() {
    let data;
    try {
      data = await (await fetch('/api/hot')).json();
    } catch {
      return; // 拿不到就退回静态建议，首页没必要为此显示一行错误
    }
    const groups = (data && data.groups) || [];
    if (!groups.length) return; // 一条都没推出来，静态建议留着更稳妥

    el.suggestions.replaceChildren();
    for (const group of groups) {
      const head = document.createElement('div');
      head.className = 'sug-group';
      head.textContent = group.label;
      if (data.stale) {
        head.classList.add('sug-stale');
        head.title = '这次没刷新成功，显示的是上一次拿到的内容';
      }
      el.suggestions.appendChild(head);
      for (const item of group.items) {
        // 问句后面必须缀上来源链接：模型手里没有联网搜索工具（那要配 API Key），
        // 只有 http_request。不给它 URL，它就只能凭记忆答，
        // 而推送的恰恰都是训练数据之后的事——那正是最容易编的地方。
        const payload =
          item.question + '\n\n来源：' + item.url + '（' + item.meta + '）';
        el.suggestions.appendChild(suggestionButton(item.question, payload));
      }
    }
  }

  function renderTools() {
    el.toolList.replaceChildren();
    for (const name of state.info.tools) {
      const li = document.createElement('li');
      li.appendChild(span('n', name));
      el.toolList.appendChild(li);
    }
  }

  async function boot() {
    fillIcons(document); // 先把静态 HTML 里那些 [data-icon] 占位换成图标
    try {
      const response = await fetch('/api/info');
      if (bouncedToLogin(response)) return;
      state.info = await response.json();
    } catch {
      el.modelText.textContent = '未连接';
      return;
    }
    state.info.settings = state.info.settings || {};
    state.sessionId = (state.info.session || {}).id || '';
    el.crumb.textContent = (state.info.session || {}).title || '新对话';
    el.modelText.textContent = `${state.info.provider} · ${state.info.model}`;
    el.chip.title = `${state.info.provider} / ${state.info.model} · 最多 ${state.info.max_steps} 步`;
    renderSuggestions();
    renderTools();
    refreshSessions();
    restoreSession();
    renderHot(); // 不 await：主页先出来，热点随后把静态建议换掉
  }

  /**
   * 把上次那个会话的历史铺出来。
   *
   * 这是「会话落盘」在界面上兑现的地方：刷新页面、重启程序之后，看到的还是
   * 上次的对话，而不是一张白纸。走 GET /api/session 而不是 POST select ——
   * 服务端本来就停在最近活动的那个会话上，读一次就行。
   */
  async function restoreSession() {
    let detail;
    try {
      detail = await getJSON('/api/session');
    } catch {
      return;
    }
    state.sessionId = detail.id;
    renderTranscript(detail);
  }

  // ---- 事件绑定 ----

  el.send.addEventListener('click', () => ask(el.input.value));

  el.stop.addEventListener('click', cancel);

  el.input.addEventListener('input', autosize);
  el.input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      ask(el.input.value);
    }
  });

  // 生成过程中按 Esc 也能停。弹着对话框时不抢——那时候 Esc 是「关掉对话框」
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && state.running && el.dialog.hidden) cancel();
  });

  $('reset-btn').addEventListener('click', reset);
  $('tools-btn').addEventListener('click', () => {
    el.sheet.hidden = false;
  });
  $('sheet-close').addEventListener('click', () => {
    el.sheet.hidden = true;
  });
  el.sheet.addEventListener('click', (event) => {
    if (event.target === el.sheet) el.sheet.hidden = true;
  });

  $('new-chat').addEventListener('click', newSession);
  $('clear-all').addEventListener('click', clearAllSessions);
  $('notice-close').addEventListener('click', () => {
    el.notice.hidden = true;
  });

  for (const tab of el.sideTabs) {
    tab.addEventListener('click', () => showView(tab.dataset.view));
  }

  // 窄屏的抽屉。scrim 是抽屉外面那层暗底，点它关掉
  $('side-show').addEventListener('click', () => {
    el.app.classList.add('side-open');
    el.scrim.hidden = false;
  });
  $('side-hide').addEventListener('click', () => {
    el.app.classList.remove('side-open');
    el.scrim.hidden = true;
  });
  el.scrim.addEventListener('click', () => {
    el.app.classList.remove('side-open');
    el.scrim.hidden = true;
  });

  // 上下文状态：指上去展开，点一下钉住（钉住之后移开鼠标不收回，方便细看）
  let ctxPinned = false;
  const showCtxDetail = (show) => {
    el.ctxDetail.hidden = !show;
    el.ctxLine.setAttribute('aria-expanded', String(show));
  };
  el.ctx.addEventListener('mouseenter', () => showCtxDetail(true));
  el.ctx.addEventListener('mouseleave', () => {
    if (!ctxPinned) showCtxDetail(false);
  });
  el.ctxLine.addEventListener('click', () => {
    ctxPinned = !ctxPinned;
    showCtxDetail(ctxPinned);
  });

  el.stream.addEventListener('scroll', () => {
    state.pinned = atBottom();
  });

  const topbar = document.querySelector('.topbar');
  el.stream.addEventListener('scroll', () => {
    topbar.classList.toggle('scrolled', el.stream.scrollTop > 4);
  });

  boot();
  el.input.focus();
})();
