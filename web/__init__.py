"""Web 入口层（设计方案 §九 第四阶段「Web UI / API 服务化」）。

和 :mod:`main` 是**平级的两个入口**：都只做「读输入 → 交给 Agent → 输出结果」，
装配那一份共用 ``main.build_agent()``，所以下面四层一行都不用为网页改。

- :mod:`web.server`  HTTP + SSE 服务，零第三方依赖
- ``web/static/``    前端三件套（HTML / CSS / JS），无构建步骤
"""
