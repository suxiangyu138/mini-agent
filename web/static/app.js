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

  const ESCAPES = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' };
  const escapeHtml = (s) => String(s).replace(/[&<>"']/g, (c) => ESCAPES[c]);

  /** 行内标记：代码、加粗、链接。**入参必须先转义**，这里只负责套标签。 */
  function inline(escaped) {
    return escaped
      .replace(/`([^`\n]+)`/g, '<code>$1</code>')
      .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
      .replace(
        /(https?:\/\/[^\s<>()]+)/g,
        '<a href="$1" target="_blank" rel="noopener noreferrer">$1</a>'
      );
  }

  /** 极简 markdown：只认围栏代码块、段落、行内代码/加粗/链接——够用就好。 */
  function renderRich(text) {
    const frag = document.createDocumentFragment();
    String(text)
      .split(/```/)
      .forEach((part, index) => {
        if (index % 2 === 1) {
          // 奇数段落是围栏里的内容，第一行可能是语言名
          const newline = part.indexOf('\n');
          const head = newline >= 0 ? part.slice(0, newline).trim() : '';
          const isLang = /^[\w+#.-]{0,16}$/.test(head);
          const body = newline >= 0 && isLang ? part.slice(newline + 1) : part;
          const pre = document.createElement('pre');
          const code = document.createElement('code');
          code.textContent = body.replace(/\n$/, '');
          pre.appendChild(code);
          frag.appendChild(pre);
          return;
        }
        part.split(/\n{2,}/).forEach((block) => {
          if (!block.trim()) return;
          const p = document.createElement('p');
          // 这里是全文件唯一一处 innerHTML，安全性由 escapeHtml 保证：
          // 先转义掉所有 & < > " '，再只加回 <code>/<strong>/<a> 三种固定标签。
          // 用户的链接也进不到属性里做坏事——引号在转义阶段已经变成实体了。
          p.innerHTML = inline(escapeHtml(block)).replace(/\n/g, '<br>');
          frag.appendChild(p);
        });
      });
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
