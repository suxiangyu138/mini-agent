/* Mini-Agent 前端
 *
 * 无框架、无构建，一个文件读完。和后端只有三个接口：
 *   GET  /api/info    界面启动时问一次当前配置和工具
 *   POST /api/chat    提问，响应是 SSE 流（text / step / done / error 四种事件）
 *   POST /api/cancel  中断正在跑的那一轮
 *
 * 一条硬规则：**模型输出的文字一律先转义再进 DOM**。
 * 模型会把 http_request 抓回来的网页内容当上下文，那里面写什么都有可能，
 * 不转义就等于把 XSS 直接送到页面上。
 */

(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);

  const el = {
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
  };

  const state = {
    info: { tools: [], suggestions: [] },
    running: false, // 正在跑一轮
    turn: null, // 当前这一轮的可变引用
    pinned: true, // 用户是否还停在底部（自己往上翻了就别硬拽回来）
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

  function makeTurn(question) {
    el.empty.hidden = true;

    const turn = document.createElement('article');
    turn.className = 'turn';

    const row = document.createElement('div');
    row.className = 'msg-user';
    row.appendChild(span('stamp', stamp()));
    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    bubble.textContent = question;
    row.appendChild(bubble);
    turn.appendChild(row);

    const bot = document.createElement('div');
    bot.className = 'msg-bot';
    bot.setAttribute('aria-busy', 'true');
    turn.appendChild(bot);

    // 流式期间有个「正在发生」的指标行垫在最底下，新内容插在它前面，
    // 这样工具块和文字块的先后顺序自然就是它们真实发生的顺序。
    const live = document.createElement('div');
    live.className = 'metrics live';
    bot.appendChild(live);

    el.turns.appendChild(turn);

    state.turn = {
      node: turn,
      bot,
      live,
      prose: null, // 当前正在追加文字的段落块
      text: '', // 已经收到的正文
      chars: 0,
      startedAt: performance.now(),
      firstAt: null,
      lastPaint: 0,
    };
    paintLive();
    scrollToEnd(true);
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

  function paintLive() {
    const t = state.turn;
    if (!t) return;
    const now = performance.now();
    if (now - t.lastPaint < 90) return; // 别为了跳数字把主线程占满
    t.lastPaint = now;
    const row = liveRow(t);
    t.live.replaceWith(row);
    t.live = row; // 引用必须跟着换到新节点上，否则下一次 replaceWith 打在已摘除的旧节点上
  }

  /** 正文段落：第一段新建，之后往同一段里追加，直到来了一次工具调用把它「封口」。 */
  function appendText(delta) {
    const t = state.turn;
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
    paintLive();
    scrollToEnd();
  }

  function sealProse() {
    const t = state.turn;
    if (!t || !t.prose) return;
    t.prose.classList.remove('streaming');
    t.prose.replaceChildren(renderRich(t.text));
    t.prose = null;
    t.text = '';
  }

  function appendTool(step) {
    const t = state.turn;
    if (!t) return;
    sealProse(); // 工具之后的文字属于新的一段，另起一块

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

    t.bot.insertBefore(box, t.live);
    scrollToEnd();
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

  function renderMetrics(metrics) {
    const t = state.turn;
    if (!t) return;
    sealProse();

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
    t.bot.setAttribute('aria-busy', 'false');
    t.node.appendChild(span('stamp', stamp()));
    scrollToEnd();
  }

  function renderError(message) {
    const t = state.turn;
    if (!t) return;
    sealProse();
    const row = document.createElement('div');
    row.className = 'metrics';
    row.appendChild(metric('', message, '', 'warn'));
    t.live.replaceWith(row);
    t.live = row;
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

  async function ask(question) {
    if (state.running || !question.trim()) return;

    makeTurn(question);
    setRunning(true);
    el.input.value = '';
    autosize();

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: question }),
      });
      if (!response.ok || !response.body) {
        renderError(`请求失败（HTTP ${response.status}）`);
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
          handleFrame(buffer.slice(0, cut));
          buffer = buffer.slice(cut + 2);
        }
      }
      sealProse();
    } catch (error) {
      renderError(`连接中断：${error.message}`);
    } finally {
      setRunning(false);
      state.turn = null;
      el.input.focus();
    }
  }

  function handleFrame(frame) {
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

    if (name === 'text') appendText(data.delta || '');
    else if (name === 'step') appendTool(data);
    else if (name === 'done') renderMetrics(data.metrics);
    else if (name === 'error') renderError(data.message || '出错了');
  }

  async function cancel() {
    try {
      await fetch('/api/cancel', { method: 'POST' });
    } catch {
      /* 连接都没了，这一轮本来也停了 */
    }
  }

  async function reset() {
    if (state.running) await cancel();
    await fetch('/api/reset', { method: 'POST' });
    el.turns.replaceChildren();
    el.empty.hidden = false;
    el.input.focus();
  }

  function autosize() {
    el.input.style.height = 'auto';
    el.input.style.height = Math.min(el.input.scrollHeight, 200) + 'px';
  }

  // ==================================================================== 启动

  function renderSuggestions() {
    el.suggestions.replaceChildren();
    for (const question of state.info.suggestions) {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = question;
      button.addEventListener('click', () => {
        el.input.value = question;
        autosize();
        ask(question);
      });
      el.suggestions.appendChild(button);
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
    try {
      state.info = await (await fetch('/api/info')).json();
    } catch {
      state.modelText.textContent = '未连接';
      return;
    }
    el.modelText.textContent = `${state.info.provider} · ${state.info.model}`;
    el.chip.title = `${state.info.provider} / ${state.info.model} · 最多 ${state.info.max_steps} 步`;
    renderSuggestions();
    renderTools();
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

  // 生成过程中按 Esc 也能停
  document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && state.running) cancel();
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
