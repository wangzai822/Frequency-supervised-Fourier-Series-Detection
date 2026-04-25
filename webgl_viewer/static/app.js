/* eslint-disable no-console */
(() => {
  "use strict";

  const VERSION = "0.6.0-native-webgl";

  const SCIENTIFIC_PALETTE = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#7F3C8D",
    "#4E79A7",
    "#59A14F",
    "#B07AA1",
    "#9C755F",
    "#6B7280",
    "#1F77B4",
    "#8C564B",
    "#2CA02C",
    "#9467BD",
    "#BCBD22",
    "#17BECF"
  ];

  const state = {
    initialized: false,
    busy: false,

    dbPath: "",
    imageRoot: "",

    summary: {},
    images: [],
    filteredImages: [],

    currentImage: null,
    currentImageIndex: -1,

    records: [],
    selectedRecordIndex: -1,
    hoverRecordIndex: -1,

    selectedDefectDetail: null,
    selectedDefectDetailLoading: false,
    coeffHoverIndex: -1,

    imageSource: null,
    imageTextureSource: null,
    lastTransform: null,

    labelBoxes: [],

    health: null,
    logs: [],
    resizeObserver: null,
    drawRaf: 0,

    viewerRenderer: null,
    coeffRenderer: null
  };

  const els = {};

  // ---------------------------------------------------------------------------
  // DOM helpers
  // ---------------------------------------------------------------------------

  function cssEscape(value) {
    if (window.CSS && typeof window.CSS.escape === "function") {
      return window.CSS.escape(String(value));
    }

    return String(value).replace(/["\\#.;,[\](){}:+~>*^$|=\s]/g, "\\$&");
  }

  function pickElement(ids) {
    for (const id of ids) {
      const el = document.getElementById(id);
      if (el) return el;
    }

    for (const id of ids) {
      const el = document.querySelector(`[data-role="${cssEscape(id)}"]`);
      if (el) return el;
    }

    return null;
  }

  function collectElements() {
    const map = {
      dbPathInput: [
        "dbPathInput",
        "dbPath",
        "databasePathInput",
        "databasePath",
        "sqlitePathInput",
        "sqlitePath"
      ],
      imageRootInput: [
        "imageRootInput",
        "imageRoot",
        "imageFolderInput",
        "imageFolder",
        "imageRepositoryInput",
        "imageRepository",
        "imageDirInput",
        "imageDir"
      ],
      selectDbBtn: [
        "selectDbBtn",
        "selectDatabaseBtn",
        "browseDbBtn",
        "pickDbBtn"
      ],
      selectFolderBtn: [
        "selectFolderBtn",
        "selectImageRootBtn",
        "browseFolderBtn",
        "browseImagesBtn",
        "pickFolderBtn"
      ],
      initializeBtn: [
        "initializeBtn",
        "initBtn",
        "loadArchiveBtn",
        "openArchiveBtn"
      ],
      refreshImagesBtn: [
        "refreshImagesBtn",
        "reloadImagesBtn",
        "refreshBtn"
      ],
      searchInput: [
        "searchInput",
        "imageSearchInput",
        "searchImagesInput",
        "filterInput"
      ],
      existingOnlyInput: [
        "existingOnlyInput",
        "onlyExistingInput",
        "existingOnly",
        "showExistingOnly"
      ],
      polygonPointsInput: [
        "polygonPointsInput",
        "polygonPoints",
        "pointsInput"
      ],
      opacityInput: [
        "opacityInput",
        "overlayOpacityInput",
        "overlayOpacity"
      ],
      lineWidthInput: [
        "lineWidthInput",
        "strokeWidthInput",
        "polygonLineWidth"
      ],
      fitModeSelect: [
        "fitModeSelect",
        "imageFitSelect",
        "fitMode"
      ],
      showLabelsInput: [
        "showLabelsInput",
        "showLabels",
        "labelToggle"
      ],
      showVerticesInput: [
        "showVerticesInput",
        "showVertices",
        "vertexToggle"
      ],
      imageList: [
        "imageList",
        "imagesList",
        "imageBrowser",
        "imageRecords"
      ],
      imageCountText: [
        "imageCountText",
        "imageListCount",
        "visibleImageCount"
      ],
      canvas: [
        "viewerCanvas",
        "webglCanvas",
        "imageCanvas",
        "overlayCanvas",
        "canvas"
      ],
      viewerLabelLayer: [
        "viewerLabelLayer",
        "labelLayer",
        "annotationLabelLayer"
      ],
      viewerPlaceholder: [
        "viewerPlaceholder",
        "canvasPlaceholder",
        "viewerEmptyText"
      ],
      sessionMessage: [
        "sessionMessage",
        "statusMessage",
        "messageText",
        "runtimeMessage"
      ],
      serverStatus: [
        "serverStatus",
        "healthStatus",
        "statusBadge"
      ],
      runtimeLog: [
        "runtimeLog",
        "logPanel",
        "eventLog"
      ],
      toastContainer: [
        "toastContainer",
        "toasts"
      ],
      summaryImages: [
        "summaryImages",
        "summaryImageCount",
        "imageCount"
      ],
      summaryDefects: [
        "summaryDefects",
        "summaryDefectCount",
        "defectCount"
      ],
      summaryClasses: [
        "summaryClasses",
        "summaryClassCount",
        "classCount"
      ],
      summaryFound: [
        "summaryFound",
        "summaryImagesFound",
        "foundImageCount"
      ],
      summaryMissing: [
        "summaryMissing",
        "summaryMissingImages",
        "missingImageCount"
      ],
      summaryDbSize: [
        "summaryDbSize",
        "summaryDatabaseSize",
        "dbSize"
      ],
      currentImageName: [
        "currentImageName",
        "selectedImageName"
      ],
      currentImagePath: [
        "currentImagePath",
        "selectedImagePath"
      ],
      currentImageSize: [
        "currentImageSize",
        "selectedImageSize"
      ],
      currentImageStatus: [
        "currentImageStatus",
        "selectedImageStatus"
      ],
      currentRecordCount: [
        "currentRecordCount",
        "selectedRecordCount"
      ],
      recordsTableBody: [
        "recordsTableBody",
        "recordTableBody",
        "annotationTableBody"
      ],
      recordsList: [
        "recordsList",
        "recordList",
        "annotationList"
      ],
      selectedDefectSection: [
        "selectedDefectSection",
        "selectedDefectPreview",
        "defectPreview"
      ],
      selectedDefectBadge: [
        "selectedDefectBadge",
        "defectPreviewBadge"
      ],
      selectedDefectTitle: [
        "selectedDefectTitle",
        "defectPreviewTitle"
      ],
      selectedDefectSubtitle: [
        "selectedDefectSubtitle",
        "defectPreviewSubtitle"
      ],
      selectedDefectClass: [
        "selectedDefectClass"
      ],
      selectedDefectScore: [
        "selectedDefectScore"
      ],
      selectedDefectArea: [
        "selectedDefectArea"
      ],
      selectedDefectPerimeter: [
        "selectedDefectPerimeter"
      ],
      selectedDefectOrientation: [
        "selectedDefectOrientation"
      ],
      selectedDefectElongation: [
        "selectedDefectElongation"
      ],
      selectedDefectId: [
        "selectedDefectId"
      ],
      selectedDefectImage: [
        "selectedDefectImage"
      ],
      selectedDefectClassId: [
        "selectedDefectClassId"
      ],
      selectedDefectCodec: [
        "selectedDefectCodec"
      ],
      selectedDefectCoeffColumn: [
        "selectedDefectCoeffColumn"
      ],
      selectedDefectCoeffCount: [
        "selectedDefectCoeffCount"
      ],
      coeffSpectrumCanvas: [
        "coeffSpectrumCanvas",
        "coefficientSpectrumCanvas"
      ],
      coeffSpectrumOverlay: [
        "coeffSpectrumOverlay",
        "coefficientSpectrumOverlay"
      ],
      coeffIndexText: [
        "coeffIndexText",
        "coefficientIndexText"
      ],
      coeffValueText: [
        "coeffValueText",
        "coefficientValueText"
      ],
      coeffHint: [
        "coeffHint",
        "coefficientHint"
      ]
    };

    for (const key of Object.keys(els)) {
      delete els[key];
    }

    for (const [key, ids] of Object.entries(map)) {
      els[key] = pickElement(ids);
    }
  }

  function setText(el, value) {
    if (!el) return;
    el.textContent = value == null ? "" : String(value);
  }

  function setValue(el, value) {
    if (!el) return;
    el.value = value == null ? "" : String(value);
  }

  function getValue(el, fallback = "") {
    if (!el) return fallback;
    return el.value == null ? fallback : String(el.value);
  }

  function getChecked(el) {
    if (!el) return false;
    return Boolean(el.checked);
  }

  function finiteNumber(value, fallback = 0) {
    if (value === null || value === undefined) return fallback;

    if (typeof value === "string" && value.trim() === "") {
      return fallback;
    }

    const number = Number(value);
    return Number.isFinite(number) ? number : fallback;
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function normalizeSlashPath(value) {
    return String(value || "").replace(/\\/g, "/");
  }

  function basename(value) {
    const clean = normalizeSlashPath(value)
      .split("?")[0]
      .replace(/\/+$/g, "");

    if (!clean) return "";
    return clean.split("/").pop() || "";
  }

  function pickFirstDefined(record, keys) {
    if (!record) return undefined;

    for (const key of keys) {
      if (!Object.prototype.hasOwnProperty.call(record, key)) {
        continue;
      }

      const value = record[key];

      if (value !== undefined && value !== null && value !== "") {
        return value;
      }
    }

    return undefined;
  }

  function boolFromUnknown(value, fallback = false) {
    if (value === undefined || value === null || value === "") {
      return fallback;
    }

    if (typeof value === "boolean") return value;
    if (typeof value === "number") return value !== 0;

    const text = String(value).trim().toLowerCase();

    if (["1", "true", "yes", "y", "on", "found", "exists"].includes(text)) {
      return true;
    }

    if (["0", "false", "no", "n", "off", "missing", "none"].includes(text)) {
      return false;
    }

    return Boolean(value);
  }

  function formatNumber(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
      return value == null || value === "" ? "--" : String(value);
    }

    return new Intl.NumberFormat().format(number);
  }

  function formatBytes(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
      return value == null || value === "" ? "--" : String(value);
    }

    const units = ["B", "KB", "MB", "GB", "TB"];
    let n = number;
    let i = 0;

    while (n >= 1024 && i < units.length - 1) {
      n /= 1024;
      i += 1;
    }

    if (i === 0) {
      return `${Math.round(n)} ${units[i]}`;
    }

    return `${n.toFixed(2)} ${units[i]}`;
  }

  function formatMetric(value, unit = "", digits = 2) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
      return "--";
    }

    const formatted = number.toLocaleString(undefined, {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits
    });

    return unit ? `${formatted} ${unit}` : formatted;
  }

  function formatScore(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
      return "--";
    }

    if (Math.abs(number) <= 1) {
      return number.toFixed(3);
    }

    return number.toFixed(2);
  }

  function formatAngle(value) {
    const number = Number(value);

    if (!Number.isFinite(number)) {
      return "--";
    }

    return `${number.toFixed(2)}°`;
  }

  function joinUrl(base, params = {}) {
    const url = new URL(base, window.location.origin);

    for (const [key, value] of Object.entries(params || {})) {
      if (value === undefined || value === null || value === "") {
        continue;
      }

      if (Array.isArray(value)) {
        for (const item of value) {
          url.searchParams.append(key, String(item));
        }
      } else {
        url.searchParams.set(key, String(value));
      }
    }

    if (url.origin === window.location.origin) {
      return `${url.pathname}${url.search}`;
    }

    return url.toString();
  }

  async function fetchJson(url, options = {}) {
    const fetchOptions = {
      ...options
    };

    const headers = new Headers(fetchOptions.headers || {});

    if (
      fetchOptions.body &&
      typeof fetchOptions.body === "object" &&
      !(fetchOptions.body instanceof FormData) &&
      !(fetchOptions.body instanceof Blob) &&
      !(fetchOptions.body instanceof ArrayBuffer)
    ) {
      if (!headers.has("Content-Type")) {
        headers.set("Content-Type", "application/json");
      }

      fetchOptions.body = JSON.stringify(fetchOptions.body);
    }

    fetchOptions.headers = headers;

    const response = await fetch(url, fetchOptions);
    const text = await response.text();

    let payload = null;

    if (text) {
      try {
        payload = JSON.parse(text);
      } catch {
        payload = text;
      }
    }

    if (!response.ok) {
      let detail = "";

      if (payload && typeof payload === "object") {
        detail = payload.detail || payload.message || JSON.stringify(payload);
      } else if (payload) {
        detail = String(payload);
      }

      throw new Error(detail || `${response.status} ${response.statusText}`);
    }

    if (payload === null) {
      return {};
    }

    return payload;
  }

  function addLog(message, type = "info") {
    const time = new Date().toLocaleTimeString();
    const line = `[${time}] ${message}`;

    state.logs.push({ time, type, message });

    if (state.logs.length > 500) {
      state.logs.shift();
    }

    if (!els.runtimeLog) {
      if (type === "error") {
        console.error(line);
      } else if (type === "warn") {
        console.warn(line);
      } else {
        console.log(line);
      }

      return;
    }

    const item = document.createElement("div");
    item.className = `log-line log-${type}`;
    item.textContent = line;

    els.runtimeLog.appendChild(item);
    els.runtimeLog.scrollTop = els.runtimeLog.scrollHeight;

    while (els.runtimeLog.children.length > 300) {
      els.runtimeLog.removeChild(els.runtimeLog.firstChild);
    }
  }

  function toast(message, type = "info", timeout = 3200) {
    if (!els.toastContainer) {
      if (type === "error") {
        console.error(message);
      } else {
        console.log(message);
      }
      return;
    }

    const item = document.createElement("div");
    item.className = `toast toast-${type}`;
    item.textContent = message;

    els.toastContainer.appendChild(item);

    window.setTimeout(() => {
      item.classList.add("toast-out");
      window.setTimeout(() => item.remove(), 240);
    }, timeout);
  }

  function setBusy(value, message = "") {
    state.busy = Boolean(value);
    document.body.classList.toggle("is-busy", state.busy);

    if (els.initializeBtn) {
      els.initializeBtn.disabled = state.busy;
    }

    if (els.refreshImagesBtn) {
      els.refreshImagesBtn.disabled = state.busy;
    }

    if (message) {
      setText(els.sessionMessage, message);
    }
  }

  // ---------------------------------------------------------------------------
  // WebGL helpers
  // ---------------------------------------------------------------------------

  function createGLContext(canvas, options = {}) {
    const attrs = {
      alpha: true,
      antialias: true,
      premultipliedAlpha: false,
      preserveDrawingBuffer: false,
      ...options
    };

    const gl =
      canvas.getContext("webgl2", attrs) ||
      canvas.getContext("webgl", attrs) ||
      canvas.getContext("experimental-webgl", attrs);

    if (!gl) {
      throw new Error("WebGL is not supported by this browser.");
    }

    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    return gl;
  }

  function compileShader(gl, type, source) {
    const shader = gl.createShader(type);

    gl.shaderSource(shader, source);
    gl.compileShader(shader);

    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader) || "Unknown shader compile error.";
      gl.deleteShader(shader);
      throw new Error(info);
    }

    return shader;
  }

  function createProgram(gl, vertexSource, fragmentSource) {
    const vertexShader = compileShader(gl, gl.VERTEX_SHADER, vertexSource);
    const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, fragmentSource);

    const program = gl.createProgram();

    gl.attachShader(program, vertexShader);
    gl.attachShader(program, fragmentShader);
    gl.linkProgram(program);

    gl.deleteShader(vertexShader);
    gl.deleteShader(fragmentShader);

    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
      const info = gl.getProgramInfoLog(program) || "Unknown WebGL program link error.";
      gl.deleteProgram(program);
      throw new Error(info);
    }

    return program;
  }

  function resizeWebGLCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    const cssWidth = Math.max(
      1,
      Math.round(rect.width || canvas.clientWidth || canvas.width || 960)
    );

    const cssHeight = Math.max(
      1,
      Math.round(rect.height || canvas.clientHeight || canvas.height || 640)
    );

    const pixelWidth = Math.max(1, Math.round(cssWidth * dpr));
    const pixelHeight = Math.max(1, Math.round(cssHeight * dpr));

    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
    }

    return {
      width: cssWidth,
      height: cssHeight,
      pixelWidth,
      pixelHeight,
      dpr
    };
  }

  function cssToGlColor(value, fallback = [1, 1, 1, 1]) {
    const text = String(value || "").trim();

    if (!text) {
      return fallback.slice();
    }

    if (text.startsWith("#")) {
      const rgba = hexToRgbaArray(text, 1);
      return rgba || fallback.slice();
    }

    const match = text.match(
      /^rgba?\s*\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*([\d.]+))?\s*\)$/i
    );

    if (match) {
      const r = clamp(Number(match[1]) / 255, 0, 1);
      const g = clamp(Number(match[2]) / 255, 0, 1);
      const b = clamp(Number(match[3]) / 255, 0, 1);
      const a = match[4] === undefined ? 1 : clamp(Number(match[4]), 0, 1);

      return [r, g, b, a];
    }

    return fallback.slice();
  }

  function hexToRgbaArray(hex, alpha = 1) {
    let clean = String(hex || "").replace("#", "").trim();

    if (/^[0-9a-f]{3}$/i.test(clean)) {
      clean = clean
        .split("")
        .map((ch) => ch + ch)
        .join("");
    }

    if (!/^[0-9a-f]{6}$/i.test(clean)) {
      return null;
    }

    const r = parseInt(clean.slice(0, 2), 16) / 255;
    const g = parseInt(clean.slice(2, 4), 16) / 255;
    const b = parseInt(clean.slice(4, 6), 16) / 255;

    return [r, g, b, clamp(alpha, 0, 1)];
  }

  function rgbaArrayToCss(color) {
    const r = Math.round(clamp(color[0], 0, 1) * 255);
    const g = Math.round(clamp(color[1], 0, 1) * 255);
    const b = Math.round(clamp(color[2], 0, 1) * 255);
    const a = clamp(color[3] ?? 1, 0, 1);

    return `rgba(${r},${g},${b},${a})`;
  }

  function multiplyAlpha(color, alpha) {
    return [
      color[0],
      color[1],
      color[2],
      clamp((color[3] ?? 1) * alpha, 0, 1)
    ];
  }

  function colorForInstanceArray(index, alpha = 1) {
    const color = SCIENTIFIC_PALETTE[
      ((index % SCIENTIFIC_PALETTE.length) + SCIENTIFIC_PALETTE.length) %
        SCIENTIFIC_PALETTE.length
    ];

    return hexToRgbaArray(color, alpha) || [1, 1, 1, alpha];
  }

  function strokeColorArrayForRecord(record, index, alpha = 1) {
    const explicit = record?.color || record?.stroke_color || record?.instance_color;

    if (explicit && typeof explicit === "string") {
      return cssToGlColor(explicit, colorForInstanceArray(index, alpha)).map((value, channel) =>
        channel === 3 ? alpha : value
      );
    }

    return colorForInstanceArray(index, alpha);
  }

  function strokeColorForRecord(record, index, alpha = 1) {
    return rgbaArrayToCss(strokeColorArrayForRecord(record, index, alpha));
  }

  // ---------------------------------------------------------------------------
  // WebGL viewer renderer
  // ---------------------------------------------------------------------------

  class WebGLViewerRenderer {
    constructor(canvas) {
      this.canvas = canvas;
      this.gl = createGLContext(canvas);
      this.imageTexture = null;
      this.imageSource = null;

      this.size = {
        width: 1,
        height: 1,
        pixelWidth: 1,
        pixelHeight: 1,
        dpr: 1
      };

      this._initPrograms();
      this._initBuffers();

      addLog("Native WebGL viewer renderer initialized.", "success");
    }

    _initPrograms() {
      const gl = this.gl;

      const vertexSource = `
        attribute vec2 a_position;
        attribute vec2 a_texcoord;

        uniform vec2 u_resolution;

        varying vec2 v_texcoord;

        void main() {
          vec2 zeroToOne = a_position / u_resolution;
          vec2 clip = zeroToOne * 2.0 - 1.0;

          gl_Position = vec4(clip * vec2(1.0, -1.0), 0.0, 1.0);
          v_texcoord = a_texcoord;
        }
      `;

      const textureFragmentSource = `
        precision mediump float;

        varying vec2 v_texcoord;

        uniform sampler2D u_texture;
        uniform float u_opacity;

        void main() {
          vec4 color = texture2D(u_texture, v_texcoord);
          gl_FragColor = vec4(color.rgb, color.a * u_opacity);
        }
      `;

      const colorFragmentSource = `
        precision mediump float;

        uniform vec4 u_color;

        void main() {
          gl_FragColor = u_color;
        }
      `;

      this.textureProgram = createProgram(gl, vertexSource, textureFragmentSource);
      this.colorProgram = createProgram(gl, vertexSource, colorFragmentSource);

      this.textureLocations = {
        position: gl.getAttribLocation(this.textureProgram, "a_position"),
        texcoord: gl.getAttribLocation(this.textureProgram, "a_texcoord"),
        resolution: gl.getUniformLocation(this.textureProgram, "u_resolution"),
        texture: gl.getUniformLocation(this.textureProgram, "u_texture"),
        opacity: gl.getUniformLocation(this.textureProgram, "u_opacity")
      };

      this.colorLocations = {
        position: gl.getAttribLocation(this.colorProgram, "a_position"),
        texcoord: gl.getAttribLocation(this.colorProgram, "a_texcoord"),
        resolution: gl.getUniformLocation(this.colorProgram, "u_resolution"),
        color: gl.getUniformLocation(this.colorProgram, "u_color")
      };
    }

    _initBuffers() {
      const gl = this.gl;

      this.positionBuffer = gl.createBuffer();
      this.texcoordBuffer = gl.createBuffer();

      gl.bindBuffer(gl.ARRAY_BUFFER, this.texcoordBuffer);
      gl.bufferData(
        gl.ARRAY_BUFFER,
        new Float32Array([
          0, 0,
          1, 0,
          0, 1,
          0, 1,
          1, 0,
          1, 1
        ]),
        gl.STATIC_DRAW
      );
    }

    resize() {
      this.size = resizeWebGLCanvas(this.canvas);

      const gl = this.gl;
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);

      return this.size;
    }

    clear() {
      const gl = this.gl;

      gl.clearColor(0.027, 0.067, 0.122, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);

      this.drawBackground();
    }

    ensureImageTexture(image) {
      if (!image) {
        return null;
      }

      if (this.imageTexture && this.imageSource === image) {
        return this.imageTexture;
      }

      const gl = this.gl;

      if (this.imageTexture) {
        gl.deleteTexture(this.imageTexture);
        this.imageTexture = null;
      }

      const texture = gl.createTexture();

      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
      gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);

      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA,
        gl.RGBA,
        gl.UNSIGNED_BYTE,
        image
      );

      this.imageTexture = texture;
      this.imageSource = image;

      return texture;
    }

    drawImage(image, rect) {
      if (!image || !rect) return;

      const texture = this.ensureImageTexture(image);

      if (!texture) return;

      const gl = this.gl;

      const x1 = rect.x;
      const y1 = rect.y;
      const x2 = rect.x + rect.width;
      const y2 = rect.y + rect.height;

      const positions = new Float32Array([
        x1, y1,
        x2, y1,
        x1, y2,
        x1, y2,
        x2, y1,
        x2, y2
      ]);

      gl.useProgram(this.textureProgram);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, positions, gl.STREAM_DRAW);

      gl.enableVertexAttribArray(this.textureLocations.position);
      gl.vertexAttribPointer(this.textureLocations.position, 2, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.texcoordBuffer);
      gl.enableVertexAttribArray(this.textureLocations.texcoord);
      gl.vertexAttribPointer(this.textureLocations.texcoord, 2, gl.FLOAT, false, 0, 0);

      gl.uniform2f(this.textureLocations.resolution, this.size.width, this.size.height);
      gl.uniform1f(this.textureLocations.opacity, 1);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.uniform1i(this.textureLocations.texture, 0);

      gl.drawArrays(gl.TRIANGLES, 0, 6);

      this.drawRectOutline(rect, [0.058, 0.09, 0.165, 0.28], 1);
    }

    drawBackground() {
      const w = this.size.width;
      const h = this.size.height;

      this.drawSolidRect(0, 0, w, h, [0.027, 0.067, 0.122, 1]);

      const grid = 32;
      const lines = [];

      for (let x = 0; x <= w; x += grid) {
        lines.push([x, 0], [x, h]);
      }

      for (let y = 0; y <= h; y += grid) {
        lines.push([0, y], [w, y]);
      }

      this.drawLineSegments(lines, [1, 1, 1, 0.075], 1);
    }

    drawSolidRect(x, y, width, height, color) {
      const points = [
        [x, y],
        [x + width, y],
        [x, y + height],
        [x, y + height],
        [x + width, y],
        [x + width, y + height]
      ];

      this.drawTriangles(points, color);
    }

    drawRectOutline(rect, color, width = 1) {
      const points = [
        [rect.x, rect.y],
        [rect.x + rect.width, rect.y],
        [rect.x + rect.width, rect.y + rect.height],
        [rect.x, rect.y + rect.height]
      ];

      this.drawPolyline(points, {
        color,
        width,
        closed: true
      });
    }

    drawTriangles(points, color) {
      if (!points || points.length < 3) return;

      const gl = this.gl;
      const data = new Float32Array(points.flat());

      gl.useProgram(this.colorProgram);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STREAM_DRAW);

      gl.enableVertexAttribArray(this.colorLocations.position);
      gl.vertexAttribPointer(this.colorLocations.position, 2, gl.FLOAT, false, 0, 0);

      gl.disableVertexAttribArray(this.colorLocations.texcoord);

      gl.uniform2f(this.colorLocations.resolution, this.size.width, this.size.height);
      gl.uniform4f(this.colorLocations.color, color[0], color[1], color[2], color[3]);

      gl.drawArrays(gl.TRIANGLES, 0, points.length);
    }

    drawLineSegments(segments, color, width = 1) {
      if (!segments || segments.length < 2) return;

      const triangles = [];

      for (let i = 0; i + 1 < segments.length; i += 2) {
        appendSegmentTriangles(triangles, segments[i], segments[i + 1], width);
      }

      this.drawTriangles(triangles, color);
    }

    drawPolyline(points, options = {}) {
      const {
        color = [1, 1, 1, 1],
        width = 1,
        closed = true
      } = options;

      if (!points || points.length < 2) return;

      const triangles = [];

      for (let i = 0; i < points.length - 1; i += 1) {
        appendSegmentTriangles(triangles, points[i], points[i + 1], width);
      }

      if (closed && points.length >= 3) {
        appendSegmentTriangles(triangles, points[points.length - 1], points[0], width);
      }

      this.drawTriangles(triangles, color);
    }

    drawCircle(cx, cy, radius, color, segments = 18) {
      const triangles = [];

      for (let i = 0; i < segments; i += 1) {
        const a0 = (Math.PI * 2 * i) / segments;
        const a1 = (Math.PI * 2 * (i + 1)) / segments;

        triangles.push(
          [cx, cy],
          [cx + Math.cos(a0) * radius, cy + Math.sin(a0) * radius],
          [cx + Math.cos(a1) * radius, cy + Math.sin(a1) * radius]
        );
      }

      this.drawTriangles(triangles, color);
    }

    drawScientificOutline(points, options = {}) {
      const {
        color = [0, 0.45, 0.7, 0.9],
        selected = false,
        hovered = false,
        closed = true,
        baseWidth = 2.25
      } = options;

      if (!points || points.length < 2) return;

      const mainWidth = selected
        ? baseWidth + 0.75
        : hovered
          ? baseWidth + 0.35
          : baseWidth;

      const casingWidth = selected
        ? mainWidth + 2.9
        : hovered
          ? mainWidth + 2.25
          : mainWidth + 1.75;

      const hairlineWidth = selected
        ? Math.max(0.85, mainWidth * 0.28)
        : hovered
          ? Math.max(0.7, mainWidth * 0.22)
          : 0;

      const casing = selected
        ? [0.058, 0.09, 0.165, 0.78]
        : hovered
          ? [0.058, 0.09, 0.165, 0.66]
          : [0.058, 0.09, 0.165, 0.52];

      this.drawPolyline(points, {
        color: casing,
        width: casingWidth,
        closed
      });

      this.drawPolyline(points, {
        color,
        width: mainWidth,
        closed
      });

      if (hairlineWidth > 0) {
        this.drawPolyline(points, {
          color: selected
            ? [0.972, 0.98, 0.988, 0.88]
            : [0.972, 0.98, 0.988, 0.58],
          width: hairlineWidth,
          closed
        });
      }
    }
  }

  function appendSegmentTriangles(out, p1, p2, width) {
    const x1 = Number(p1[0]);
    const y1 = Number(p1[1]);
    const x2 = Number(p2[0]);
    const y2 = Number(p2[1]);

    if (![x1, y1, x2, y2].every(Number.isFinite)) return;

    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.hypot(dx, dy);

    if (len <= 0.0001) return;

    const nx = -dy / len;
    const ny = dx / len;
    const half = Math.max(0.25, width / 2);

    const ax = x1 + nx * half;
    const ay = y1 + ny * half;
    const bx = x1 - nx * half;
    const by = y1 - ny * half;
    const cx = x2 + nx * half;
    const cy = y2 + ny * half;
    const dx2 = x2 - nx * half;
    const dy2 = y2 - ny * half;

    out.push(
      [ax, ay],
      [bx, by],
      [cx, cy],
      [cx, cy],
      [bx, by],
      [dx2, dy2]
    );
  }

  // ---------------------------------------------------------------------------
  // WebGL coefficient spectrum renderer
  // ---------------------------------------------------------------------------

  class WebGLSpectrumRenderer {
    constructor(canvas) {
      this.canvas = canvas;
      this.gl = createGLContext(canvas);
      this.size = {
        width: 1,
        height: 1,
        pixelWidth: 1,
        pixelHeight: 1,
        dpr: 1
      };

      this._initProgram();
      this.positionBuffer = this.gl.createBuffer();

      addLog("Native WebGL coefficient spectrum renderer initialized.", "success");
    }

    _initProgram() {
      const vertexSource = `
        attribute vec2 a_position;

        uniform vec2 u_resolution;

        void main() {
          vec2 zeroToOne = a_position / u_resolution;
          vec2 clip = zeroToOne * 2.0 - 1.0;

          gl_Position = vec4(clip * vec2(1.0, -1.0), 0.0, 1.0);
        }
      `;

      const fragmentSource = `
        precision mediump float;

        uniform vec4 u_color;

        void main() {
          gl_FragColor = u_color;
        }
      `;

      const gl = this.gl;

      this.program = createProgram(gl, vertexSource, fragmentSource);
      this.locations = {
        position: gl.getAttribLocation(this.program, "a_position"),
        resolution: gl.getUniformLocation(this.program, "u_resolution"),
        color: gl.getUniformLocation(this.program, "u_color")
      };
    }

    resize() {
      this.size = resizeWebGLCanvas(this.canvas);

      const gl = this.gl;
      gl.viewport(0, 0, this.canvas.width, this.canvas.height);

      return this.size;
    }

    clear() {
      const gl = this.gl;

      gl.clearColor(0.039, 0.078, 0.141, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
    }

    drawTriangles(points, color) {
      if (!points || points.length < 3) return;

      const gl = this.gl;
      const data = new Float32Array(points.flat());

      gl.useProgram(this.program);

      gl.bindBuffer(gl.ARRAY_BUFFER, this.positionBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STREAM_DRAW);

      gl.enableVertexAttribArray(this.locations.position);
      gl.vertexAttribPointer(this.locations.position, 2, gl.FLOAT, false, 0, 0);

      gl.uniform2f(this.locations.resolution, this.size.width, this.size.height);
      gl.uniform4f(this.locations.color, color[0], color[1], color[2], color[3]);

      gl.drawArrays(gl.TRIANGLES, 0, points.length);
    }

    drawRect(x, y, width, height, color) {
      this.drawTriangles(
        [
          [x, y],
          [x + width, y],
          [x, y + height],
          [x, y + height],
          [x + width, y],
          [x + width, y + height]
        ],
        color
      );
    }

    drawLine(x1, y1, x2, y2, color, width = 1) {
      const triangles = [];
      appendSegmentTriangles(triangles, [x1, y1], [x2, y2], width);
      this.drawTriangles(triangles, color);
    }

    drawRoundedBar(x, y, width, height, color) {
      this.drawRect(x, y, width, height, color);
    }

    drawSpectrum(coefficients, hoverIndex) {
      this.resize();
      this.clear();

      const width = this.size.width;
      const height = this.size.height;

      this.drawRect(0, 0, width, height, [0.039, 0.078, 0.141, 1]);

      for (let x = 0; x < width; x += 28) {
        this.drawLine(x, 0, x, height, [0.58, 0.64, 0.72, 0.07], 1);
      }

      for (let y = 0; y < height; y += 28) {
        this.drawLine(0, y, width, y, [0.58, 0.64, 0.72, 0.07], 1);
      }

      if (!coefficients.length) {
        return;
      }

      const paddingLeft = 12;
      const paddingRight = 12;
      const paddingTop = 12;
      const paddingBottom = 18;

      const plotW = Math.max(1, width - paddingLeft - paddingRight);
      const plotH = Math.max(1, height - paddingTop - paddingBottom);

      const minValue = Math.min(...coefficients);
      const maxValue = Math.max(...coefficients);
      const maxAbs = Math.max(...coefficients.map((value) => Math.abs(value)), 1e-9);

      const hasNegative = minValue < 0;
      const hasPositive = maxValue > 0;

      let zeroY;

      if (hasNegative && hasPositive) {
        zeroY = paddingTop + (maxValue / (maxValue - minValue)) * plotH;
      } else if (hasNegative) {
        zeroY = paddingTop;
      } else {
        zeroY = paddingTop + plotH;
      }

      this.drawLine(
        paddingLeft,
        zeroY,
        width - paddingRight,
        zeroY,
        [1, 1, 1, 0.36],
        1
      );

      const count = coefficients.length;
      const slot = plotW / count;
      const barW = Math.max(2, Math.min(14, slot * 0.72));

      for (let i = 0; i < count; i += 1) {
        const value = Number(coefficients[i]);

        if (!Number.isFinite(value)) {
          continue;
        }

        const abs = Math.abs(value);
        const barH = Math.max(1, (abs / maxAbs) * plotH * 0.92);

        const x = paddingLeft + i * slot + (slot - barW) / 2;
        const y = value >= 0 ? zeroY - barH : zeroY;
        const h = barH;

        const color = i === hoverIndex
          ? [0.98, 0.8, 0.08, 0.96]
          : value >= 0
            ? [0.13, 0.75, 0.78, 0.88]
            : [0.86, 0.22, 0.55, 0.88];

        this.drawRoundedBar(x, y, barW, h, color);

        if (i === hoverIndex) {
          this.drawLine(x, y, x + barW, y, [1, 1, 1, 0.8], 1.1);
          this.drawLine(x, y + h, x + barW, y + h, [1, 1, 1, 0.8], 1.1);
          this.drawLine(x, y, x, y + h, [1, 1, 1, 0.8], 1.1);
          this.drawLine(x + barW, y, x + barW, y + h, [1, 1, 1, 0.8], 1.1);
        }
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Coefficient helpers
  // ---------------------------------------------------------------------------

  function flattenNumeric(value, out = [], depth = 0) {
    if (depth > 8) {
      return out;
    }

    if (value === null || value === undefined || value === "") {
      return out;
    }

    if (typeof value === "number") {
      if (Number.isFinite(value)) {
        out.push(value);
      }
      return out;
    }

    if (typeof value === "string") {
      const text = value.trim();

      if (!text) {
        return out;
      }

      try {
        return flattenNumeric(JSON.parse(text), out, depth + 1);
      } catch {
        const nums =
          text.match(/-?\d+(?:\.\d+)?(?:e[-+]?\d+)?/gi)?.map(Number) || [];

        for (const num of nums) {
          if (Number.isFinite(num)) {
            out.push(num);
          }
        }

        return out;
      }
    }

    if (Array.isArray(value)) {
      for (const item of value) {
        flattenNumeric(item, out, depth + 1);
      }
      return out;
    }

    if (typeof value === "object") {
      const preferredKeys = [
        "coefficients",
        "fourier_coefficients",
        "fourier_coeffs",
        "fourier_coef",
        "coeffs",
        "values",
        "data",
        "spectrum",
        "descriptor",
        "real",
        "imag",
        "x",
        "y"
      ];

      let usedPreferred = false;

      for (const key of preferredKeys) {
        if (Object.prototype.hasOwnProperty.call(value, key)) {
          flattenNumeric(value[key], out, depth + 1);
          usedPreferred = true;
        }
      }

      if (!usedPreferred) {
        for (const item of Object.values(value)) {
          flattenNumeric(item, out, depth + 1);
        }
      }

      return out;
    }

    return out;
  }

  function extractCoefficientArray(value) {
    const numbers = flattenNumeric(value, [], 0);

    return numbers
      .map(Number)
      .filter((number) => Number.isFinite(number));
  }

  function getRecordCoefficientArray(record) {
    if (!record) return [];

    if (Array.isArray(record.coefficients) && record.coefficients.length) {
      return record.coefficients
        .map(Number)
        .filter((number) => Number.isFinite(number));
    }

    const raw = pickFirstDefined(record, [
      "coefficients",
      "fourier_coefficients",
      "fourier_coeffs",
      "fourier_coef",
      "coeffs",
      "coeff",
      "descriptor",
      "descriptor_spectrum",
      "spectrum",
      "fourier_descriptor",
      "fourier_descriptors"
    ]);

    return extractCoefficientArray(raw);
  }

  // ---------------------------------------------------------------------------
  // Normalizers
  // ---------------------------------------------------------------------------

  function normalizeImageRecord(input, index = 0) {
    const record =
      input && typeof input === "object"
        ? input
        : {
            value: input
          };

    const file =
      pickFirstDefined(record, [
        "source_path",
        "resolved_path",
        "file",
        "filename",
        "file_name",
        "image_file",
        "image_filename",
        "relative_path",
        "path",
        "image_path"
      ]) ?? "";

    const imageId =
      pickFirstDefined(record, [
        "image_id",
        "id",
        "imageId",
        "database_image_id"
      ]) ?? "";

    const key =
      pickFirstDefined(record, [
        "image_key",
        "key",
        "uuid",
        "uid",
        "id",
        "image_id"
      ]) ??
      imageId ??
      file ??
      `image-${index + 1}`;

    const name =
      pickFirstDefined(record, [
        "name",
        "filename",
        "file_name",
        "image_name",
        "display_name"
      ]) ||
      basename(file) ||
      String(imageId || key || `image-${index + 1}`);

    const existsValue = pickFirstDefined(record, [
      "image_exists",
      "exists",
      "found",
      "available",
      "is_found"
    ]);

    const exists = boolFromUnknown(
      existsValue,
      Boolean(record.resolved_path || record.image_exists === true || record.exists === true)
    );

    const defectCountValue =
      finiteNumber(
        pickFirstDefined(record, [
          "det_count",
          "defects",
          "defectCount",
          "defect_count",
          "damage_count",
          "record_count",
          "annotations",
          "annotation_count",
          "instances",
          "instance_count"
        ]),
        NaN
      );

    const defectCount = Number.isFinite(defectCountValue) ? defectCountValue : 0;

    const widthValue =
      finiteNumber(
        pickFirstDefined(record, [
          "img_w",
          "width",
          "image_width",
          "w"
        ]),
        NaN
      );

    const heightValue =
      finiteNumber(
        pickFirstDefined(record, [
          "img_h",
          "height",
          "image_height",
          "h"
        ]),
        NaN
      );

    const width = Number.isFinite(widthValue) ? widthValue : 0;
    const height = Number.isFinite(heightValue) ? heightValue : 0;

    const sourcePath = String(
      record.source_path ||
        record.path ||
        record.image_path ||
        file ||
        ""
    );

    const resolvedPath = String(record.resolved_path || "");

    return {
      ...record,

      __index: index,

      key: String(key),
      image_key: String(record.image_key || key),
      id: imageId || key,
      image_id: imageId || record.id || name || key,

      name: String(name),
      file: String(file || ""),
      path: String(record.source_path || record.path || record.image_path || file || ""),
      source_path: sourcePath,
      resolved_path: resolvedPath,

      exists,
      image_exists: exists,

      defectCount,
      det_count: defectCount,
      defect_count: defectCount,

      width,
      height,
      img_w: width,
      img_h: height
    };
  }

  function normalizePointList(value) {
    if (value === undefined || value === null || value === "") {
      return [];
    }

    if (typeof value === "string") {
      const text = value.trim();

      if (!text) {
        return [];
      }

      try {
        return normalizePointList(JSON.parse(text));
      } catch {
        const nums =
          text.match(/-?\d+(?:\.\d+)?(?:e[-+]?\d+)?/gi)?.map(Number) || [];

        const points = [];

        for (let i = 0; i + 1 < nums.length; i += 2) {
          const x = nums[i];
          const y = nums[i + 1];

          if (Number.isFinite(x) && Number.isFinite(y)) {
            points.push([x, y]);
          }
        }

        return points;
      }
    }

    if (Array.isArray(value)) {
      if (value.length === 0) {
        return [];
      }

      if (Array.isArray(value[0])) {
        return value
          .map((point) => [Number(point[0]), Number(point[1])])
          .filter((point) => Number.isFinite(point[0]) && Number.isFinite(point[1]));
      }

      if (typeof value[0] === "object" && value[0] !== null) {
        return value
          .map((point) => [
            Number(point.x ?? point.X ?? point[0]),
            Number(point.y ?? point.Y ?? point[1])
          ])
          .filter((point) => Number.isFinite(point[0]) && Number.isFinite(point[1]));
      }

      if (typeof value[0] === "number") {
        const points = [];

        for (let i = 0; i + 1 < value.length; i += 2) {
          const x = Number(value[i]);
          const y = Number(value[i + 1]);

          if (Number.isFinite(x) && Number.isFinite(y)) {
            points.push([x, y]);
          }
        }

        return points;
      }
    }

    if (typeof value === "object") {
      if (Array.isArray(value.points)) {
        return normalizePointList(value.points);
      }

      if (Array.isArray(value.polygon)) {
        return normalizePointList(value.polygon);
      }

      if (Array.isArray(value.x) && Array.isArray(value.y)) {
        const count = Math.min(value.x.length, value.y.length);
        const points = [];

        for (let i = 0; i < count; i += 1) {
          const x = Number(value.x[i]);
          const y = Number(value.y[i]);

          if (Number.isFinite(x) && Number.isFinite(y)) {
            points.push([x, y]);
          }
        }

        return points;
      }
    }

    return [];
  }

  function normalizeBBox(value, record = {}) {
    if (value === undefined || value === null || value === "") {
      value = null;
    }

    if (typeof value === "string" && value.trim()) {
      try {
        return normalizeBBox(JSON.parse(value), record);
      } catch {
        const nums =
          value.match(/-?\d+(?:\.\d+)?(?:e[-+]?\d+)?/gi)?.map(Number) || [];

        if (nums.length >= 4) {
          return {
            x: nums[0],
            y: nums[1],
            width: nums[2],
            height: nums[3]
          };
        }
      }
    }

    if (Array.isArray(value) && value.length >= 4) {
      return {
        x: Number(value[0]),
        y: Number(value[1]),
        width: Number(value[2]),
        height: Number(value[3])
      };
    }

    if (value && typeof value === "object") {
      const x = Number(value.x ?? value.left ?? value.xmin ?? value.x_min ?? value.min_x);
      const y = Number(value.y ?? value.top ?? value.ymin ?? value.y_min ?? value.min_y);

      if (
        value.width !== undefined ||
        value.w !== undefined ||
        value.height !== undefined ||
        value.h !== undefined
      ) {
        return {
          x,
          y,
          width: Number(value.width ?? value.w),
          height: Number(value.height ?? value.h)
        };
      }

      const xMax = Number(value.xmax ?? value.x_max ?? value.max_x ?? value.right);
      const yMax = Number(value.ymax ?? value.y_max ?? value.max_y ?? value.bottom);

      if (
        Number.isFinite(x) &&
        Number.isFinite(y) &&
        Number.isFinite(xMax) &&
        Number.isFinite(yMax)
      ) {
        return {
          x,
          y,
          width: xMax - x,
          height: yMax - y
        };
      }
    }

    const x = Number(
      record.x ??
        record.left ??
        record.xmin ??
        record.x_min ??
        record.min_x
    );

    const y = Number(
      record.y ??
        record.top ??
        record.ymin ??
        record.y_min ??
        record.min_y
    );

    const w = Number(record.width ?? record.w ?? record.bbox_w ?? record.box_w);
    const h = Number(record.height ?? record.h ?? record.bbox_h ?? record.box_h);

    if (
      Number.isFinite(x) &&
      Number.isFinite(y) &&
      Number.isFinite(w) &&
      Number.isFinite(h)
    ) {
      return {
        x,
        y,
        width: w,
        height: h
      };
    }

    const xMax = Number(record.xmax ?? record.x_max ?? record.max_x ?? record.right);
    const yMax = Number(record.ymax ?? record.y_max ?? record.max_y ?? record.bottom);

    if (
      Number.isFinite(x) &&
      Number.isFinite(y) &&
      Number.isFinite(xMax) &&
      Number.isFinite(yMax)
    ) {
      return {
        x,
        y,
        width: xMax - x,
        height: yMax - y
      };
    }

    return null;
  }

  function normalizeDamageRecord(input, index = 0) {
    const record =
      input && typeof input === "object"
        ? input
        : {
            value: input
          };

    const id =
      pickFirstDefined(record, [
        "defect_id",
        "record_id",
        "annotation_id",
        "damage_id",
        "id",
        "uuid"
      ]) ?? `record-${index + 1}`;

    const className =
      pickFirstDefined(record, [
        "class_name",
        "class",
        "damage_type",
        "label",
        "category",
        "category_name",
        "type"
      ]) ?? "damage";

    const polygon = normalizePointList(
      pickFirstDefined(record, [
        "polygon",
        "polygon_points",
        "points",
        "contour",
        "vertices",
        "boundary"
      ])
    );

    const bbox = normalizeBBox(
      pickFirstDefined(record, [
        "bbox",
        "box",
        "bounding_box",
        "rect",
        "rectangle"
      ]),
      record
    );

    const area = finiteNumber(
      pickFirstDefined(record, [
        "area",
        "area_px2",
        "pixel_area",
        "polygon_area"
      ]),
      NaN
    );

    const perimeter = finiteNumber(
      pickFirstDefined(record, [
        "perimeter",
        "perimeter_px",
        "length",
        "arc_length"
      ]),
      NaN
    );

    const orientation = finiteNumber(
      pickFirstDefined(record, [
        "orientation",
        "orientation_deg",
        "angle",
        "theta"
      ]),
      NaN
    );

    const elongation = finiteNumber(
      pickFirstDefined(record, [
        "elongation",
        "eccentricity",
        "aspect_ratio",
        "ratio"
      ]),
      NaN
    );

    const score = finiteNumber(
      pickFirstDefined(record, [
        "score",
        "confidence",
        "conf",
        "probability",
        "class_score"
      ]),
      NaN
    );

    const classId =
      pickFirstDefined(record, [
        "class_id",
        "category_id",
        "label_id",
        "damage_class_id"
      ]) ?? "";

    const codec =
      pickFirstDefined(record, [
        "codec",
        "coeff_codec",
        "fourier_codec",
        "encoding",
        "descriptor_codec"
      ]) ?? "";

    const coeffColumn =
      pickFirstDefined(record, [
        "coeff_column",
        "coefficient_column",
        "fourier_column",
        "coeff_col",
        "descriptor_column"
      ]) ?? "";

    const coefficients = getRecordCoefficientArray(record);

    return {
      ...record,

      __index: index,

      id: String(id),
      defect_id: String(record.defect_id || id),
      record_id: String(record.record_id || id),

      class_name: String(className),
      class: String(className),
      damage_type: String(record.damage_type || className),

      polygon,
      polygon_points: polygon,
      bbox,

      area,
      area_px2: area,

      perimeter,
      perimeter_px: perimeter,

      orientation,
      orientation_deg: orientation,

      elongation,

      score,
      confidence: score,

      class_id: classId,
      codec,
      coeff_column: coeffColumn,

      coefficients
    };
  }

  function extractImages(payload) {
    if (Array.isArray(payload)) {
      return payload;
    }

    if (!payload || typeof payload !== "object") {
      return [];
    }

    for (const key of ["images", "items", "records", "data", "image_records"]) {
      if (Array.isArray(payload[key])) {
        return payload[key];
      }
    }

    return [];
  }

  function extractRecords(payload) {
    if (Array.isArray(payload)) {
      return payload;
    }

    if (!payload || typeof payload !== "object") {
      return [];
    }

    for (const key of ["records", "defects", "annotations", "items", "data"]) {
      if (Array.isArray(payload[key])) {
        return payload[key];
      }
    }

    return [];
  }

  function getImageKey(image) {
    if (!image) return "";

    return String(
      image.image_key ||
        image.key ||
        image.id ||
        image.image_id ||
        image.source_path ||
        image.path ||
        image.file ||
        ""
    );
  }

  function imageSearchText(image) {
    return [
      image.name,
      image.image_key,
      image.key,
      image.image_id,
      image.id,
      image.source_path,
      image.resolved_path,
      image.path,
      image.file
    ]
      .filter(Boolean)
      .join(" ")
      .toLowerCase();
  }

  function getRecordStableId(record) {
    if (!record) return "";

    return String(
      record.defect_id ||
        record.damage_id ||
        record.annotation_id ||
        record.record_id ||
        record.id ||
        ""
    );
  }

  function getFetchableDefectId(record) {
    if (!record) return "";

    const explicit =
      record.defect_id ||
      record.damage_id ||
      record.annotation_id ||
      "";

    if (explicit) {
      return String(explicit);
    }

    const generic = String(record.id || record.record_id || "");

    if (!generic || /^record-\d+$/i.test(generic)) {
      return "";
    }

    return generic;
  }

  // ---------------------------------------------------------------------------
  // API actions
  // ---------------------------------------------------------------------------

  async function checkHealth() {
    try {
      const payload = await fetchJson("/api/health");

      state.health = payload;

      const ready = Boolean(payload.ready || payload.db_loaded);

      setText(
        els.serverStatus,
        ready ? "Ready" : "Idle"
      );

      if (els.serverStatus) {
        els.serverStatus.classList.toggle("is-ready", ready);
        els.serverStatus.classList.toggle("is-idle", !ready);
        els.serverStatus.classList.remove("is-error");
      }

      return payload;
    } catch (error) {
      setText(els.serverStatus, "Offline");

      if (els.serverStatus) {
        els.serverStatus.classList.remove("is-ready");
        els.serverStatus.classList.remove("is-idle");
        els.serverStatus.classList.add("is-error");
      }

      addLog(`Health check failed: ${error.message}`, "error");
      return null;
    }
  }

  async function selectDatabase() {
    try {
      setBusy(true, "Opening database picker...");
      addLog("Opening database picker...");

      const payload = await fetchJson("/api/select-db", {
        method: "POST"
      });

      const path =
        payload.path ||
        payload.db_path ||
        payload.database_path ||
        payload.sqlite_path ||
        "";

      if (path) {
        setValue(els.dbPathInput, path);
        state.dbPath = path;
        addLog(`Selected database: ${path}`);
        toast("数据库已选择。", "success");
      } else {
        addLog("Database picker cancelled.", "warn");
      }
    } catch (error) {
      addLog(`Database picker failed: ${error.message}`, "error");
      toast(`数据库选择失败：${error.message}`, "error", 4800);
    } finally {
      setBusy(false);
    }
  }

  async function selectImageFolder() {
    try {
      setBusy(true, "Opening image repository picker...");
      addLog("Opening image repository picker...");

      const payload = await fetchJson("/api/select-folder", {
        method: "POST"
      });

      const path =
        payload.path ||
        payload.folder ||
        payload.directory ||
        payload.image_root ||
        payload.image_folder ||
        payload.image_repository ||
        "";

      if (path) {
        setValue(els.imageRootInput, path);
        state.imageRoot = path;
        addLog(`Selected image repository: ${path}`);
        toast("图像目录已选择。", "success");
      } else {
        addLog("Image folder picker cancelled.", "warn");
      }
    } catch (error) {
      addLog(`Image folder picker failed: ${error.message}`, "error");
      toast(`图像目录选择失败：${error.message}`, "error", 4800);
    } finally {
      setBusy(false);
    }
  }

  async function initializeArchive() {
    const dbPath = getValue(els.dbPathInput).trim();
    const imageRoot = getValue(els.imageRootInput).trim();

    if (!dbPath) {
      toast("请先选择 SQLite 数据库。", "warn");
      setText(els.sessionMessage, "Please select a SQLite database first.");
      return;
    }

    state.dbPath = dbPath;
    state.imageRoot = imageRoot;

    try {
      setBusy(true, "Initializing archive...");
      setText(els.sessionMessage, "Initializing archive...");
      addLog(`Initializing archive: ${dbPath}`);

      const payload = await fetchJson("/api/initialize", {
        method: "POST",
        body: {
          db_path: dbPath,
          image_root: imageRoot
        }
      });

      state.initialized = true;
      state.summary = payload.summary || {};
      state.images = extractImages(payload).map((image, index) =>
        normalizeImageRecord(image, index)
      );

      state.currentImage = null;
      state.currentImageIndex = -1;
      state.records = [];
      state.selectedRecordIndex = -1;
      state.hoverRecordIndex = -1;
      state.selectedDefectDetail = null;
      state.selectedDefectDetailLoading = false;
      state.coeffHoverIndex = -1;
      state.imageSource = null;
      state.imageTextureSource = null;
      state.labelBoxes = [];

      updateSummary(state.summary);
      renderImages();
      renderCurrentImagePanel();
      renderRecords();
      scheduleDraw();

      const imageCount = state.images.length;
      const defectCount =
        state.summary.defect_count ||
        state.summary.defects ||
        state.summary.total_defects ||
        0;

      setText(
        els.sessionMessage,
        `Archive initialized. ${formatNumber(imageCount)} images, ${formatNumber(defectCount)} defects.`
      );

      addLog(
        `Archive initialized successfully: ${formatNumber(imageCount)} images, ${formatNumber(defectCount)} defects.`,
        "success"
      );

      toast("初始化完成。", "success");

      if (state.images.length > 0) {
        await selectImage(state.images[0]);
      }

      await checkHealth();
    } catch (error) {
      const message = error?.message || String(error);

      setText(els.sessionMessage, `Initialization failed: ${message}`);
      addLog(`Archive initialization failed: ${message}`, "error");
      toast(`初始化失败：${message.slice(0, 160)}`, "error", 5200);
    } finally {
      setBusy(false);
    }
  }

  async function loadImages() {
    try {
      setBusy(true, "Loading image list...");

      const payload = await fetchJson(
        joinUrl("/api/images", {
          page: 1,
          page_size: 50000
        })
      );

      state.summary = payload.summary || state.summary || {};
      state.images = extractImages(payload).map((image, index) =>
        normalizeImageRecord(image, index)
      );

      updateSummary(state.summary);
      renderImages();

      addLog(`Loaded ${state.images.length} images.`);
      toast("图像列表已刷新。", "success");
    } catch (error) {
      addLog(`Failed to load images: ${error.message}`, "error");
      toast(`图像列表加载失败：${error.message}`, "error", 4800);
    } finally {
      setBusy(false);
    }
  }

  async function loadImageForRecord(image) {
    if (!image) return null;

    state.imageSource = null;
    state.imageTextureSource = null;
    scheduleDraw();

    try {
      const src = joinUrl("/api/image", {
        key: getImageKey(image),
        image_key: image.image_key || getImageKey(image),
        image_id: image.image_id || image.id || "",
        path: image.source_path || image.path || image.file || "",
        name: image.name || "",
        _: Date.now()
      });

      const img = await loadImageElement(src);

      if (state.currentImage && getImageKey(state.currentImage) === getImageKey(image)) {
        state.imageSource = img;
        state.imageTextureSource = img;

        if (!Number.isFinite(Number(image.width)) || Number(image.width) <= 0) {
          image.width = img.naturalWidth || img.width || 0;
          image.img_w = image.width;
        }

        if (!Number.isFinite(Number(image.height)) || Number(image.height) <= 0) {
          image.height = img.naturalHeight || img.height || 0;
          image.img_h = image.height;
        }

        renderCurrentImagePanel();
        scheduleDraw();
      }

      return img;
    } catch (error) {
      state.imageSource = null;
      state.imageTextureSource = null;
      addLog(`Image load failed: ${image.name || getImageKey(image)} - ${error.message}`, "error");
      toast(`图像加载失败：${image.name || getImageKey(image)}`, "error", 3600);
      scheduleDraw();
      return null;
    }
  }

  function loadImageElement(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();

      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error(`Cannot load image from ${src}`));

      img.src = src;
    });
  }

  async function loadRecordsForImage(image) {
    if (!image) return [];

    try {
      const polygonPoints = finiteNumber(els.polygonPointsInput?.value, 256);

      const payload = await fetchJson(
        joinUrl("/api/records", {
          key: getImageKey(image),
          image_key: image.image_key || getImageKey(image),
          image_id: image.image_id || image.id || "",
          path: image.source_path || image.path || image.file || "",
          name: image.name || "",
          polygon_points: polygonPoints
        })
      );

      const records = extractRecords(payload).map((record, index) =>
        normalizeDamageRecord(record, index)
      );

      if (state.currentImage && getImageKey(state.currentImage) === getImageKey(image)) {
        state.records = records;
        state.selectedRecordIndex = records.length > 0 ? 0 : -1;
        state.hoverRecordIndex = -1;
        state.selectedDefectDetail = null;
        state.selectedDefectDetailLoading = false;
        state.coeffHoverIndex = -1;
        state.labelBoxes = [];

        renderCurrentImagePanel();
        renderRecords();
        scheduleDraw();

        if (records.length > 0) {
          loadDefectDetailForSelected(records[0], 0);
        }
      }

      addLog(`Loaded ${records.length} records for ${image.name || getImageKey(image)}.`);

      return records;
    } catch (error) {
      state.records = [];
      state.selectedRecordIndex = -1;
      state.hoverRecordIndex = -1;
      state.selectedDefectDetail = null;
      state.selectedDefectDetailLoading = false;
      state.coeffHoverIndex = -1;
      state.labelBoxes = [];

      renderCurrentImagePanel();
      renderRecords();
      scheduleDraw();

      addLog(`Record load failed: ${image.name || getImageKey(image)} - ${error.message}`, "error");
      toast(`标注记录加载失败：${error.message}`, "error", 4200);

      return [];
    }
  }

  async function loadDefectDetailForSelected(record, index) {
    const id = getFetchableDefectId(record);

    if (!id) {
      state.selectedDefectDetail = null;
      state.selectedDefectDetailLoading = false;
      renderSelectedDefectPreview();
      return;
    }

    const selectedKey = getRecordStableId(record);

    state.selectedDefectDetailLoading = true;
    renderSelectedDefectPreview();

    try {
      const polygonPoints = finiteNumber(els.polygonPointsInput?.value, 256);

      const payload = await fetchJson(
        joinUrl(`/api/defect/${encodeURIComponent(id)}`, {
          polygon_points: polygonPoints,
          include_coefficients: 1
        })
      );

      const rawDetail =
        payload.record ||
        payload.defect ||
        payload.data ||
        payload;

      const detail = normalizeDamageRecord(rawDetail, index);

      if (
        state.selectedRecordIndex === index &&
        state.records[index] &&
        getRecordStableId(state.records[index]) === selectedKey
      ) {
        const existing = state.records[index];

        const merged = {
          ...existing,
          ...detail
        };

        if (existing.polygon?.length && !detail.polygon?.length) {
          merged.polygon = existing.polygon;
          merged.polygon_points = existing.polygon;
        }

        if (existing.coefficients?.length && !detail.coefficients?.length) {
          merged.coefficients = existing.coefficients;
        }

        state.records[index] = normalizeDamageRecord(merged, index);
        state.selectedDefectDetail = state.records[index];

        renderRecords();
        scheduleDraw();
      }
    } catch (error) {
      addLog(`Defect detail unavailable for ${id}: ${error.message}`, "warn");
    } finally {
      if (state.selectedRecordIndex === index) {
        state.selectedDefectDetailLoading = false;
        renderSelectedDefectPreview();
      }
    }
  }

  async function reloadCurrentRecords() {
    if (!state.currentImage) return;

    await loadRecordsForImage(state.currentImage);
  }

  // ---------------------------------------------------------------------------
  // Rendering: summary / images / records
  // ---------------------------------------------------------------------------

  function updateSummary(summary = {}) {
    const images =
      pickFirstDefined(summary, [
        "images",
        "image_count",
        "total_images",
        "image_total"
      ]) ?? state.images.length;

    const defects =
      pickFirstDefined(summary, [
        "defects",
        "defect_count",
        "total_defects",
        "damage_count",
        "record_count"
      ]) ?? "--";

    const classes =
      pickFirstDefined(summary, [
        "classes",
        "class_count",
        "total_classes",
        "category_count"
      ]) ?? "--";

    const dbSize =
      pickFirstDefined(summary, [
        "db_size",
        "db_file_size",
        "database_size",
        "db_bytes",
        "size",
        "file_size"
      ]) ?? "--";

    const found =
      pickFirstDefined(summary, [
        "found",
        "images_found",
        "image_found_count",
        "existing",
        "existing_images",
        "available_images"
      ]) ??
      state.images.filter((image) => image.exists).length;

    const missing =
      pickFirstDefined(summary, [
        "missing",
        "missing_images",
        "missing_image_count",
        "not_found",
        "unavailable_images"
      ]) ??
      state.images.filter((image) => !image.exists).length;

    setText(els.summaryImages, formatNumber(images));
    setText(els.summaryDefects, formatNumber(defects));
    setText(els.summaryClasses, formatNumber(classes));
    setText(els.summaryFound, formatNumber(found));
    setText(els.summaryMissing, formatNumber(missing));
    setText(els.summaryDbSize, formatBytes(dbSize));
  }

  function renderImages() {
    if (!els.imageList) return;

    const query = getValue(els.searchInput).trim().toLowerCase();
    const onlyExisting = getChecked(els.existingOnlyInput);

    const filtered = state.images.filter((image) => {
      if (onlyExisting && !image.exists) {
        return false;
      }

      if (query && !imageSearchText(image).includes(query)) {
        return false;
      }

      return true;
    });

    state.filteredImages = filtered;

    setText(
      els.imageCountText,
      `${formatNumber(filtered.length)} / ${formatNumber(state.images.length)}`
    );

    els.imageList.innerHTML = "";

    if (!filtered.length) {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.textContent = state.images.length
        ? "No images match current filter."
        : "No images loaded.";
      els.imageList.appendChild(empty);
      return;
    }

    const fragment = document.createDocumentFragment();

    for (const image of filtered) {
      fragment.appendChild(createImageRow(image));
    }

    els.imageList.appendChild(fragment);
  }

  function createImageRow(image) {
    const row = document.createElement("button");
    row.type = "button";
    row.className = "image-row";

    const selected =
      state.currentImage &&
      getImageKey(state.currentImage) === getImageKey(image);

    if (selected) {
      row.classList.add("is-selected");
    }

    if (!image.exists) {
      row.classList.add("is-missing");
    }

    row.dataset.key = getImageKey(image);

    const title = document.createElement("div");
    title.className = "image-row-title";
    title.textContent = image.name || basename(image.path) || getImageKey(image);

    const meta = document.createElement("div");
    meta.className = "image-row-meta";

    const status = document.createElement("span");
    status.className = image.exists ? "badge badge-ok" : "badge badge-missing";
    status.textContent = image.exists ? "FOUND" : "MISSING";

    const count = document.createElement("span");
    count.className = "badge badge-count";
    count.textContent = `${formatNumber(image.det_count ?? image.defectCount ?? 0)} defects`;

    meta.appendChild(status);
    meta.appendChild(count);

    const path = document.createElement("div");
    path.className = "image-row-path";
    path.textContent = normalizeSlashPath(image.source_path || image.path || image.file || "");

    row.appendChild(title);
    row.appendChild(meta);
    row.appendChild(path);

    row.addEventListener("click", () => {
      selectImage(image);
    });

    return row;
  }

  async function selectImage(image) {
    if (!image) return;

    state.currentImage = image;
    state.currentImageIndex = state.images.findIndex(
      (item) => getImageKey(item) === getImageKey(image)
    );

    state.records = [];
    state.selectedRecordIndex = -1;
    state.hoverRecordIndex = -1;
    state.selectedDefectDetail = null;
    state.selectedDefectDetailLoading = false;
    state.coeffHoverIndex = -1;
    state.imageSource = null;
    state.imageTextureSource = null;
    state.labelBoxes = [];

    renderImages();
    renderCurrentImagePanel();
    renderRecords();
    scheduleDraw();

    setText(
      els.sessionMessage,
      `Loading ${image.name || getImageKey(image)}...`
    );

    const imagePromise = loadImageForRecord(image);
    const recordPromise = loadRecordsForImage(image);

    await Promise.allSettled([imagePromise, recordPromise]);

    setText(
      els.sessionMessage,
      `${image.name || getImageKey(image)} loaded.`
    );
  }

  function selectRelativeImage(offset) {
    const list = state.filteredImages.length ? state.filteredImages : state.images;

    if (!list.length) return;

    if (!state.currentImage) {
      selectImage(list[0]);
      return;
    }

    const currentKey = getImageKey(state.currentImage);
    const currentIndex = list.findIndex((image) => getImageKey(image) === currentKey);
    const nextIndex = clamp(
      currentIndex < 0 ? 0 : currentIndex + offset,
      0,
      list.length - 1
    );

    selectImage(list[nextIndex]);
  }

  function renderCurrentImagePanel() {
    const image = state.currentImage;

    if (!image) {
      setText(els.currentImageName, "--");
      setText(els.currentImagePath, "--");
      setText(els.currentImageSize, "--");
      setText(els.currentImageStatus, "--");
      setText(els.currentRecordCount, "--");
      return;
    }

    const filePath = normalizeSlashPath(
      image.source_path ||
        image.file ||
        image.path ||
        image.image_path ||
        image.resolved_path ||
        ""
    );

    const sourceSize = state.imageSource
      ? imageSize(state.imageSource)
      : {
          width: Number.isFinite(Number(image.width || image.img_w))
            ? Number(image.width || image.img_w)
            : 0,
          height: Number.isFinite(Number(image.height || image.img_h))
            ? Number(image.height || image.img_h)
            : 0
        };

    const recordCount =
      state.records.length ||
      image.defectCount ||
      image.det_count ||
      finiteNumber(
        pickFirstDefined(image, [
          "det_count",
          "defects",
          "defect_count",
          "damage_count",
          "record_count",
          "annotation_count"
        ]),
        0
      );

    setText(els.currentImageName, image.name || basename(filePath) || getImageKey(image));
    setText(els.currentImagePath, filePath || "--");

    if (sourceSize.width && sourceSize.height) {
      setText(els.currentImageSize, `${sourceSize.width} × ${sourceSize.height}`);
    } else {
      setText(els.currentImageSize, "--");
    }

    setText(els.currentImageStatus, image.exists ? "FOUND" : "MISSING");
    setText(els.currentRecordCount, formatNumber(recordCount));

    if (els.currentImageStatus) {
      els.currentImageStatus.classList.toggle("is-ready", Boolean(image.exists));
      els.currentImageStatus.classList.toggle("is-error", !image.exists);
    }
  }

  function renderRecords() {
    renderRecordTable();
    renderRecordList();
    renderSelectedDefectPreview();
  }

  function renderRecordTable() {
    if (!els.recordsTableBody) return;

    els.recordsTableBody.innerHTML = "";

    if (!state.records.length) {
      const tr = document.createElement("tr");
      const td = document.createElement("td");

      td.colSpan = 7;
      td.className = "empty-cell";
      td.textContent = "No records for current image.";

      tr.appendChild(td);
      els.recordsTableBody.appendChild(tr);
      return;
    }

    const fragment = document.createDocumentFragment();

    state.records.forEach((record, index) => {
      const tr = document.createElement("tr");
      tr.className = "record-row";
      tr.tabIndex = 0;
      tr.style.setProperty("--record-color", strokeColorForRecord(record, index, 1));

      if (index === state.selectedRecordIndex) {
        tr.classList.add("is-selected");
      }

      tr.addEventListener("click", () => {
        selectRecord(index);
      });

      tr.addEventListener("keydown", (event) => {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          selectRecord(index);
        }
      });

      const colorTd = document.createElement("td");
      colorTd.className = "record-color-cell";

      const swatch = document.createElement("span");
      swatch.className = "record-color-swatch";
      swatch.style.background = strokeColorForRecord(record, index, 1);

      colorTd.appendChild(swatch);
      tr.appendChild(colorTd);

      appendCell(tr, index + 1);
      appendCell(tr, record.class_name || record.damage_type || "--");
      appendCell(tr, formatScore(record.score));
      appendCell(tr, Number.isFinite(record.area) ? formatMetric(record.area, "px²", 0) : "--");
      appendCell(tr, Number.isFinite(record.perimeter) ? formatMetric(record.perimeter, "px", 0) : "--");
      appendCell(tr, record.polygon?.length || 0);

      fragment.appendChild(tr);
    });

    els.recordsTableBody.appendChild(fragment);
  }

  function appendCell(row, value) {
    const td = document.createElement("td");
    td.textContent = String(value);
    row.appendChild(td);
  }

  function renderRecordList() {
    if (!els.recordsList) return;

    els.recordsList.innerHTML = "";

    if (!state.records.length) {
      const empty = document.createElement("div");
      empty.className = "empty-state";
      empty.textContent = "No records for current image.";
      els.recordsList.appendChild(empty);
      return;
    }

    const fragment = document.createDocumentFragment();

    state.records.forEach((record, index) => {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "record-card";
      item.style.setProperty("--record-color", strokeColorForRecord(record, index, 1));

      if (index === state.selectedRecordIndex) {
        item.classList.add("is-selected");
      }

      item.addEventListener("click", () => selectRecord(index));

      const swatch = document.createElement("span");
      swatch.className = "record-color-swatch";

      const title = document.createElement("div");
      title.className = "record-card-title";
      title.textContent = `${index + 1}. ${record.class_name || record.damage_type || "damage"}`;

      const meta = document.createElement("div");
      meta.className = "record-card-meta";
      meta.textContent = `${record.polygon?.length || 0} polygon points`;

      item.appendChild(swatch);
      item.appendChild(title);
      item.appendChild(meta);

      fragment.appendChild(item);
    });

    els.recordsList.appendChild(fragment);
  }

  function selectRecord(index) {
    if (index < 0 || index >= state.records.length) {
      state.selectedRecordIndex = -1;
      state.selectedDefectDetail = null;
      state.selectedDefectDetailLoading = false;
      state.coeffHoverIndex = -1;
    } else {
      state.selectedRecordIndex = index;
      state.selectedDefectDetail = null;
      state.selectedDefectDetailLoading = false;
      state.coeffHoverIndex = -1;
    }

    renderRecords();
    scheduleDraw();

    const record = state.records[state.selectedRecordIndex];

    if (record) {
      loadDefectDetailForSelected(record, state.selectedRecordIndex);
    }
  }

  function getSelectedPreviewRecord() {
    if (
      state.selectedRecordIndex < 0 ||
      state.selectedRecordIndex >= state.records.length
    ) {
      return null;
    }

    const base = state.records[state.selectedRecordIndex];

    if (!state.selectedDefectDetail) {
      return base;
    }

    const merged = {
      ...base,
      ...state.selectedDefectDetail
    };

    if (base.polygon?.length && !state.selectedDefectDetail.polygon?.length) {
      merged.polygon = base.polygon;
      merged.polygon_points = base.polygon;
    }

    if (base.coefficients?.length && !state.selectedDefectDetail.coefficients?.length) {
      merged.coefficients = base.coefficients;
    }

    return merged;
  }

  function renderSelectedDefectPreview() {
    const record = getSelectedPreviewRecord();

    if (!els.selectedDefectSection) {
      return;
    }

    els.selectedDefectSection.classList.toggle("is-empty", !record);

    if (!record) {
      setText(els.selectedDefectBadge, "None");
      setText(els.selectedDefectTitle, "No defect selected");
      setText(els.selectedDefectSubtitle, "Click a damage record or polygon to inspect details.");

      setText(els.selectedDefectClass, "--");
      setText(els.selectedDefectScore, "--");
      setText(els.selectedDefectArea, "--");
      setText(els.selectedDefectPerimeter, "--");
      setText(els.selectedDefectOrientation, "--");
      setText(els.selectedDefectElongation, "--");
      setText(els.selectedDefectId, "--");
      setText(els.selectedDefectImage, "--");
      setText(els.selectedDefectClassId, "--");
      setText(els.selectedDefectCodec, "--");
      setText(els.selectedDefectCoeffColumn, "--");
      setText(els.selectedDefectCoeffCount, "--");

      drawCoefficientSpectrum([]);
      return;
    }

    const index = state.selectedRecordIndex;
    const color = strokeColorForRecord(record, index, 1);
    const className = record.class_name || record.damage_type || record.class || "damage";
    const coefficients = getRecordCoefficientArray(record);

    els.selectedDefectSection.style.setProperty("--selected-defect-color", color);

    setText(
      els.selectedDefectBadge,
      state.selectedDefectDetailLoading ? "Loading..." : "Selected"
    );

    setText(
      els.selectedDefectTitle,
      `${className} · Fourier coefficients`
    );

    setText(
      els.selectedDefectSubtitle,
      coefficients.length
        ? `${coefficients.length} coefficients · WebGL outline polygon display`
        : "No coefficient data returned by current record."
    );

    setText(els.selectedDefectClass, className);
    setText(els.selectedDefectScore, formatScore(record.score));
    setText(
      els.selectedDefectArea,
      Number.isFinite(record.area) ? formatMetric(record.area, "px²", 2) : "--"
    );
    setText(
      els.selectedDefectPerimeter,
      Number.isFinite(record.perimeter) ? formatMetric(record.perimeter, "px", 2) : "--"
    );
    setText(els.selectedDefectOrientation, formatAngle(record.orientation));
    setText(
      els.selectedDefectElongation,
      Number.isFinite(record.elongation) ? Number(record.elongation).toFixed(3) : "--"
    );

    setText(els.selectedDefectId, record.defect_id || record.id || "--");
    setText(els.selectedDefectImage, state.currentImage?.name || state.currentImage?.image_id || "--");
    setText(els.selectedDefectClassId, record.class_id || "--");
    setText(els.selectedDefectCodec, record.codec || "--");
    setText(els.selectedDefectCoeffColumn, record.coeff_column || "--");
    setText(els.selectedDefectCoeffCount, coefficients.length ? `${coefficients.length}` : "--");

    if (state.coeffHoverIndex >= coefficients.length) {
      state.coeffHoverIndex = -1;
    }

    updateCoeffReadout(coefficients);
    drawCoefficientSpectrum(coefficients);
  }

  // ---------------------------------------------------------------------------
  // WebGL scene drawing
  // ---------------------------------------------------------------------------

  function imageSize(source) {
    if (!source) {
      return {
        width: 0,
        height: 0
      };
    }

    return {
      width: finiteNumber(
        source.naturalWidth ??
          source.videoWidth ??
          source.width ??
          source.img_w,
        0
      ),
      height: finiteNumber(
        source.naturalHeight ??
          source.videoHeight ??
          source.height ??
          source.img_h,
        0
      )
    };
  }

  function ensureViewerRenderer() {
    if (state.viewerRenderer) {
      return state.viewerRenderer;
    }

    if (!els.canvas) {
      return null;
    }

    state.viewerRenderer = new WebGLViewerRenderer(els.canvas);
    return state.viewerRenderer;
  }

  function ensureCoeffRenderer() {
    if (state.coeffRenderer) {
      return state.coeffRenderer;
    }

    if (!els.coeffSpectrumCanvas) {
      return null;
    }

    state.coeffRenderer = new WebGLSpectrumRenderer(els.coeffSpectrumCanvas);
    return state.coeffRenderer;
  }

  function scheduleDraw() {
    if (state.drawRaf) {
      window.cancelAnimationFrame(state.drawRaf);
    }

    state.drawRaf = window.requestAnimationFrame(() => {
      state.drawRaf = 0;
      drawScene();
    });
  }

  function setViewerPlaceholder(message, visible) {
    if (!els.viewerPlaceholder) return;

    els.viewerPlaceholder.textContent = message || "";
    els.viewerPlaceholder.classList.toggle("is-visible", Boolean(visible));
  }

  function drawScene() {
    const renderer = ensureViewerRenderer();

    if (!renderer) return;

    try {
      const size = renderer.resize();

      renderer.clear();
      state.labelBoxes = [];
      clearLabelLayer();

      if (!state.imageSource) {
        state.lastTransform = null;
        setCanvasCursor(false);
        setViewerPlaceholder("Select and initialize an archive", true);
        return;
      }

      const imgSize = imageSize(state.imageSource);

      if (!imgSize.width || !imgSize.height) {
        state.lastTransform = null;
        setCanvasCursor(false);
        setViewerPlaceholder("Image size unavailable", true);
        return;
      }

      setViewerPlaceholder("", false);

      const fitMode = getValue(els.fitModeSelect, "contain") || "contain";
      const rect = computeImageRect(size.width, size.height, imgSize.width, imgSize.height, fitMode);

      state.lastTransform = {
        ...rect,
        imgW: imgSize.width,
        imgH: imgSize.height
      };

      renderer.drawImage(state.imageSource, rect);
      drawAllRecords(renderer);
    } catch (error) {
      addLog(`WebGL draw failed: ${error.message}`, "error");
      setViewerPlaceholder(`WebGL render failed: ${error.message}`, true);
    }
  }

  function computeImageRect(canvasW, canvasH, imgW, imgH, fitMode = "contain") {
    const margin = 16;
    const availableW = Math.max(1, canvasW - margin * 2);
    const availableH = Math.max(1, canvasH - margin * 2);

    let scale;

    if (fitMode === "cover") {
      scale = Math.max(availableW / imgW, availableH / imgH);
    } else if (fitMode === "original") {
      scale = 1;

      if (imgW > availableW || imgH > availableH) {
        scale = Math.min(availableW / imgW, availableH / imgH);
      }
    } else {
      scale = Math.min(availableW / imgW, availableH / imgH);
    }

    const width = imgW * scale;
    const height = imgH * scale;

    return {
      x: (canvasW - width) / 2,
      y: (canvasH - height) / 2,
      width,
      height,
      scaleX: width / imgW,
      scaleY: height / imgH
    };
  }

  function drawAllRecords(renderer) {
    const records = state.records || [];

    if (!records.length || !state.lastTransform) {
      return;
    }

    state.labelBoxes = [];
    clearLabelLayer();

    const selectedIndex = state.selectedRecordIndex;
    const hoverIndex = state.hoverRecordIndex;

    records.forEach((record, index) => {
      if (index !== selectedIndex) {
        drawRecord(renderer, record, index, false, index === hoverIndex);
      }
    });

    if (
      selectedIndex >= 0 &&
      selectedIndex < records.length
    ) {
      drawRecord(renderer, records[selectedIndex], selectedIndex, true, false);
    }

    if (getChecked(els.showLabelsInput)) {
      records.forEach((record, index) => {
        if (index !== selectedIndex) {
          drawRecordLabel(renderer, record, index, false, index === hoverIndex);
        }
      });

      if (
        selectedIndex >= 0 &&
        selectedIndex < records.length
      ) {
        drawRecordLabel(renderer, records[selectedIndex], selectedIndex, true, false);
      }
    }
  }

  function drawRecord(renderer, record, index, selected, hovered = false) {
    const transform = state.lastTransform;

    if (!transform) return;

    const polygon = Array.isArray(record.polygon)
      ? record.polygon
      : normalizePointList(record.polygon_points);

    const stroke = strokeColorArrayForRecord(
      record,
      index,
      selected ? 0.98 : hovered ? 0.94 : 0.86
    );

    if (polygon.length >= 2) {
      const points = mapPolygonToCanvas(polygon, transform);

      if (points.length < 2) {
        return;
      }

      renderer.drawScientificOutline(points, {
        color: stroke,
        selected,
        hovered,
        closed: points.length >= 3,
        baseWidth: getConfiguredLineWidth()
      });

      if (getChecked(els.showVerticesInput)) {
        drawVertices(renderer, points, selected, stroke);
      }

      return;
    }

    if (record.bbox) {
      drawBBox(renderer, record, index, selected, hovered);
    }
  }

  function drawBBox(renderer, record, index, selected, hovered = false) {
    const transform = state.lastTransform;
    const bbox = record.bbox;

    if (!transform || !bbox) return;

    const points = bboxToCanvasPoints(bbox, transform);

    if (!points.length) return;

    const stroke = strokeColorArrayForRecord(
      record,
      index,
      selected ? 0.98 : hovered ? 0.94 : 0.86
    );

    renderer.drawScientificOutline(points, {
      color: stroke,
      selected,
      hovered,
      closed: true,
      baseWidth: getConfiguredLineWidth()
    });
  }

  function getConfiguredLineWidth() {
    return clamp(finiteNumber(els.lineWidthInput?.value, 2.25), 0.75, 8);
  }

  function polygonLooksNormalized(points) {
    if (!points.length) return false;

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

    for (const point of points) {
      minX = Math.min(minX, point[0]);
      minY = Math.min(minY, point[1]);
      maxX = Math.max(maxX, point[0]);
      maxY = Math.max(maxY, point[1]);
    }

    return (
      minX >= -0.05 &&
      minY >= -0.05 &&
      maxX <= 1.5 &&
      maxY <= 1.5
    );
  }

  function mapPolygonToCanvas(points, transform) {
    const normalized = polygonLooksNormalized(points);

    return points
      .map((point) => {
        let x = Number(point[0]);
        let y = Number(point[1]);

        if (!Number.isFinite(x) || !Number.isFinite(y)) {
          return null;
        }

        if (normalized) {
          x *= transform.imgW;
          y *= transform.imgH;
        }

        return [
          transform.x + x * transform.scaleX,
          transform.y + y * transform.scaleY
        ];
      })
      .filter(Boolean);
  }

  function bboxToCanvasPoints(bbox, transform) {
    if (!bbox || !transform) return [];

    let x = Number(bbox.x);
    let y = Number(bbox.y);
    let w = Number(bbox.width);
    let h = Number(bbox.height);

    if (![x, y, w, h].every(Number.isFinite)) {
      return [];
    }

    const normalized =
      Math.abs(x) <= 1.5 &&
      Math.abs(y) <= 1.5 &&
      Math.abs(w) <= 1.5 &&
      Math.abs(h) <= 1.5;

    if (normalized) {
      x *= transform.imgW;
      y *= transform.imgH;
      w *= transform.imgW;
      h *= transform.imgH;
    }

    const canvasX = transform.x + x * transform.scaleX;
    const canvasY = transform.y + y * transform.scaleY;
    const canvasW = w * transform.scaleX;
    const canvasH = h * transform.scaleY;

    return [
      [canvasX, canvasY],
      [canvasX + canvasW, canvasY],
      [canvasX + canvasW, canvasY + canvasH],
      [canvasX, canvasY + canvasH]
    ];
  }

  function getCanvasPointsForRecord(record) {
    if (!record || !state.lastTransform) return [];

    const polygon = Array.isArray(record.polygon)
      ? record.polygon
      : normalizePointList(record.polygon_points);

    if (polygon.length >= 2) {
      return mapPolygonToCanvas(polygon, state.lastTransform);
    }

    if (record.bbox) {
      return bboxToCanvasPoints(record.bbox, state.lastTransform);
    }

    return [];
  }

  function drawVertices(renderer, points, selected, color) {
    const radius = selected ? 3.1 : 2.1;

    for (const point of points) {
      renderer.drawCircle(point[0], point[1], radius + 1, [0.058, 0.09, 0.165, 0.72], 18);
      renderer.drawCircle(
        point[0],
        point[1],
        radius,
        selected ? [1, 1, 1, 0.96] : color,
        18
      );
    }
  }

  function boundsOfPoints(points) {
    if (!points.length) {
      return {
        x: 0,
        y: 0,
        width: 0,
        height: 0,
        minX: 0,
        minY: 0,
        maxX: 0,
        maxY: 0,
        cx: 0,
        cy: 0
      };
    }

    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;

    for (const point of points) {
      minX = Math.min(minX, point[0]);
      minY = Math.min(minY, point[1]);
      maxX = Math.max(maxX, point[0]);
      maxY = Math.max(maxY, point[1]);
    }

    return {
      x: minX,
      y: minY,
      width: maxX - minX,
      height: maxY - minY,
      minX,
      minY,
      maxX,
      maxY,
      cx: (minX + maxX) / 2,
      cy: (minY + maxY) / 2
    };
  }

  function clearLabelLayer() {
    if (!els.viewerLabelLayer) return;
    els.viewerLabelLayer.innerHTML = "";
  }

  function drawRecordLabel(renderer, record, index, selected, hovered = false) {
    const points = getCanvasPointsForRecord(record);

    if (!points.length) return;

    const className = record.class_name || record.damage_type || record.class || "damage";
    const colorArray = strokeColorArrayForRecord(record, index, 1);

    drawScientificLabel(renderer, {
      points,
      index,
      text: `#${index + 1} ${className}`,
      colorArray,
      selected,
      hovered
    });
  }

  function drawScientificLabel(renderer, options) {
    const {
      points,
      index,
      text,
      colorArray,
      selected = false,
      hovered = false
    } = options;

    if (!points.length || !els.viewerLabelLayer) return;

    const canvasW = renderer.size.width;
    const canvasH = renderer.size.height;

    const fittedText = fitPlainText(text, 28);
    const labelW = Math.max(56, Math.min(220, 36 + fittedText.length * 6.4));
    const labelH = 22;

    const polygonBox = boundsOfPoints(points);
    const placement = chooseLabelPlacement({
      canvasW,
      canvasH,
      labelW,
      labelH,
      polygonBox
    });

    if (!placement) {
      return;
    }

    const {
      x,
      y,
      anchor
    } = placement;

    const labelBox = {
      x,
      y,
      width: labelW,
      height: labelH,
      index
    };

    state.labelBoxes.push(labelBox);

    const edgePoint = nearestPointOnRect(labelBox, anchor[0], anchor[1]);

    renderer.drawPolyline(
      [
        [anchor[0], anchor[1]],
        [edgePoint[0], edgePoint[1]]
      ],
      {
        color: selected
          ? multiplyAlpha(colorArray, 0.72)
          : hovered
            ? multiplyAlpha(colorArray, 0.58)
            : [0.058, 0.09, 0.165, 0.42],
        width: selected ? 1.15 : 0.9,
        closed: false
      }
    );

    const label = document.createElement("div");
    label.className = "viewer-label";

    if (selected) {
      label.classList.add("is-selected");
    }

    if (hovered) {
      label.classList.add("is-hovered");
    }

    label.style.left = `${x}px`;
    label.style.top = `${y}px`;
    label.style.width = `${labelW}px`;
    label.style.setProperty("--label-color", rgbaArrayToCss(colorArray));
    label.dataset.index = String(index);

    const dot = document.createElement("span");
    dot.className = "viewer-label-dot";

    const span = document.createElement("span");
    span.className = "viewer-label-text";
    span.textContent = fittedText;

    label.appendChild(dot);
    label.appendChild(span);

    els.viewerLabelLayer.appendChild(label);
  }

  function chooseLabelPlacement(options) {
    const {
      canvasW,
      canvasH,
      labelW,
      labelH,
      polygonBox
    } = options;

    const margin = 8;
    const offset = 10;

    const candidates = [
      {
        name: "right-top",
        x: polygonBox.maxX + offset,
        y: polygonBox.minY - labelH - 4,
        anchor: [polygonBox.maxX, polygonBox.minY]
      },
      {
        name: "left-top",
        x: polygonBox.minX - labelW - offset,
        y: polygonBox.minY - labelH - 4,
        anchor: [polygonBox.minX, polygonBox.minY]
      },
      {
        name: "right",
        x: polygonBox.maxX + offset,
        y: polygonBox.cy - labelH / 2,
        anchor: [polygonBox.maxX, polygonBox.cy]
      },
      {
        name: "left",
        x: polygonBox.minX - labelW - offset,
        y: polygonBox.cy - labelH / 2,
        anchor: [polygonBox.minX, polygonBox.cy]
      },
      {
        name: "top",
        x: polygonBox.cx - labelW / 2,
        y: polygonBox.minY - labelH - offset,
        anchor: [polygonBox.cx, polygonBox.minY]
      },
      {
        name: "bottom",
        x: polygonBox.cx - labelW / 2,
        y: polygonBox.maxY + offset,
        anchor: [polygonBox.cx, polygonBox.maxY]
      },
      {
        name: "right-bottom",
        x: polygonBox.maxX + offset,
        y: polygonBox.maxY + 4,
        anchor: [polygonBox.maxX, polygonBox.maxY]
      },
      {
        name: "left-bottom",
        x: polygonBox.minX - labelW - offset,
        y: polygonBox.maxY + 4,
        anchor: [polygonBox.minX, polygonBox.maxY]
      }
    ];

    let best = null;

    for (const candidate of candidates) {
      const rawX = candidate.x;
      const rawY = candidate.y;

      const x = clamp(rawX, margin, Math.max(margin, canvasW - labelW - margin));
      const y = clamp(rawY, margin, Math.max(margin, canvasH - labelH - margin));

      const rect = {
        x,
        y,
        width: labelW,
        height: labelH
      };

      let score = 0;

      const polygonRect = {
        x: polygonBox.x,
        y: polygonBox.y,
        width: polygonBox.width,
        height: polygonBox.height
      };

      const polygonOverlap = rectIntersectionArea(rect, polygonRect);

      if (polygonOverlap > 0) {
        score += 200000 + polygonOverlap * 40;
      }

      for (const oldBox of state.labelBoxes) {
        const overlap = rectIntersectionArea(rect, oldBox);

        if (overlap > 0) {
          score += 120000 + overlap * 60;
        }
      }

      score += Math.abs(x - rawX) * 40;
      score += Math.abs(y - rawY) * 40;

      const centerX = x + labelW / 2;
      const centerY = y + labelH / 2;

      score += Math.hypot(centerX - polygonBox.cx, centerY - polygonBox.cy) * 0.25;

      if (!best || score < best.score) {
        best = {
          ...candidate,
          x,
          y,
          score
        };
      }
    }

    return best;
  }

  function rectIntersectionArea(a, b) {
    const x1 = Math.max(a.x, b.x);
    const y1 = Math.max(a.y, b.y);
    const x2 = Math.min(a.x + a.width, b.x + b.width);
    const y2 = Math.min(a.y + a.height, b.y + b.height);

    if (x2 <= x1 || y2 <= y1) {
      return 0;
    }

    return (x2 - x1) * (y2 - y1);
  }

  function nearestPointOnRect(rect, x, y) {
    const clampedX = clamp(x, rect.x, rect.x + rect.width);
    const clampedY = clamp(y, rect.y, rect.y + rect.height);

    const candidates = [
      [clampedX, rect.y],
      [clampedX, rect.y + rect.height],
      [rect.x, clampedY],
      [rect.x + rect.width, clampedY]
    ];

    let best = candidates[0];
    let bestDistance = Infinity;

    for (const point of candidates) {
      const distance = Math.hypot(point[0] - x, point[1] - y);

      if (distance < bestDistance) {
        best = point;
        bestDistance = distance;
      }
    }

    return best;
  }

  function fitPlainText(text, maxChars) {
    const str = String(text || "");

    if (str.length <= maxChars) {
      return str;
    }

    return `${str.slice(0, Math.max(1, maxChars - 1))}…`;
  }

  // ---------------------------------------------------------------------------
  // Canvas hit testing: click polygon to select
  // ---------------------------------------------------------------------------

  function canvasPointFromEvent(event) {
    const canvas = els.canvas;

    if (!canvas) {
      return {
        x: 0,
        y: 0
      };
    }

    const rect = canvas.getBoundingClientRect();

    return {
      x: event.clientX - rect.left,
      y: event.clientY - rect.top
    };
  }

  function handleCanvasClick(event) {
    if (!state.records.length || !state.lastTransform) return;

    const point = canvasPointFromEvent(event);

    const labelHit = findLabelHitAtPoint(point.x, point.y);
    const polygonHit = labelHit >= 0
      ? labelHit
      : findHitRecordAtPoint(point.x, point.y);

    if (polygonHit >= 0) {
      event.preventDefault();
      selectRecord(polygonHit);
    }
  }

  function handleCanvasMouseMove(event) {
    if (!state.records.length || !state.lastTransform) {
      if (state.hoverRecordIndex !== -1) {
        state.hoverRecordIndex = -1;
        setCanvasCursor(false);
        scheduleDraw();
      }
      return;
    }

    const point = canvasPointFromEvent(event);

    const labelHit = findLabelHitAtPoint(point.x, point.y);
    const polygonHit = labelHit >= 0
      ? labelHit
      : findHitRecordAtPoint(point.x, point.y);

    if (polygonHit !== state.hoverRecordIndex) {
      state.hoverRecordIndex = polygonHit;
      setCanvasCursor(polygonHit >= 0);
      scheduleDraw();
    }
  }

  function handleCanvasMouseLeave() {
    if (state.hoverRecordIndex !== -1) {
      state.hoverRecordIndex = -1;
      setCanvasCursor(false);
      scheduleDraw();
    }
  }

  function setCanvasCursor(pointer) {
    if (!els.canvas) return;

    els.canvas.style.cursor = pointer ? "pointer" : "default";
  }

  function findLabelHitAtPoint(x, y) {
    for (let i = state.labelBoxes.length - 1; i >= 0; i -= 1) {
      const box = state.labelBoxes[i];

      if (
        x >= box.x &&
        x <= box.x + box.width &&
        y >= box.y &&
        y <= box.y + box.height
      ) {
        return box.index;
      }
    }

    return -1;
  }

  function findHitRecordAtPoint(x, y) {
    if (!state.lastTransform || !state.records.length) {
      return -1;
    }

    const lineTolerance = Math.max(6, getConfiguredLineWidth() + 5);

    const insideCandidates = [];
    const lineCandidates = [];

    for (let index = 0; index < state.records.length; index += 1) {
      const record = state.records[index];
      const points = getCanvasPointsForRecord(record);

      if (points.length < 2) {
        continue;
      }

      const closed = points.length >= 3;
      const distance = distanceToPolyline(x, y, points, closed);
      const lineHit = distance <= lineTolerance;
      const inside = closed ? pointInPolygon(x, y, points) : false;
      const area = closed ? Math.abs(polygonArea(points)) : Infinity;

      if (inside) {
        insideCandidates.push({
          index,
          area,
          distance
        });
      } else if (lineHit) {
        lineCandidates.push({
          index,
          area,
          distance
        });
      }
    }

    if (insideCandidates.length) {
      insideCandidates.sort((a, b) => {
        if (Math.abs(a.area - b.area) > 1) {
          return a.area - b.area;
        }

        if (Math.abs(a.distance - b.distance) > 0.5) {
          return a.distance - b.distance;
        }

        return b.index - a.index;
      });

      return insideCandidates[0].index;
    }

    if (lineCandidates.length) {
      lineCandidates.sort((a, b) => {
        if (Math.abs(a.distance - b.distance) > 0.5) {
          return a.distance - b.distance;
        }

        if (Math.abs(a.area - b.area) > 1) {
          return a.area - b.area;
        }

        return b.index - a.index;
      });

      return lineCandidates[0].index;
    }

    return -1;
  }

  function distanceToPolyline(x, y, points, closed = true) {
    if (!points.length) return Infinity;

    let best = Infinity;

    for (let i = 0; i < points.length - 1; i += 1) {
      best = Math.min(
        best,
        distancePointToSegment(
          x,
          y,
          points[i][0],
          points[i][1],
          points[i + 1][0],
          points[i + 1][1]
        )
      );
    }

    if (closed && points.length >= 3) {
      const last = points[points.length - 1];
      const first = points[0];

      best = Math.min(
        best,
        distancePointToSegment(
          x,
          y,
          last[0],
          last[1],
          first[0],
          first[1]
        )
      );
    }

    return best;
  }

  function distancePointToSegment(px, py, x1, y1, x2, y2) {
    const dx = x2 - x1;
    const dy = y2 - y1;

    if (dx === 0 && dy === 0) {
      return Math.hypot(px - x1, py - y1);
    }

    const t = clamp(
      ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy),
      0,
      1
    );

    const x = x1 + t * dx;
    const y = y1 + t * dy;

    return Math.hypot(px - x, py - y);
  }

  function pointInPolygon(x, y, points) {
    let inside = false;

    for (let i = 0, j = points.length - 1; i < points.length; j = i, i += 1) {
      const xi = points[i][0];
      const yi = points[i][1];
      const xj = points[j][0];
      const yj = points[j][1];

      const intersects =
        yi > y !== yj > y &&
        x < ((xj - xi) * (y - yi)) / ((yj - yi) || 1e-12) + xi;

      if (intersects) {
        inside = !inside;
      }
    }

    return inside;
  }

  function polygonArea(points) {
    if (!points || points.length < 3) {
      return 0;
    }

    let area = 0;

    for (let i = 0; i < points.length; i += 1) {
      const j = (i + 1) % points.length;
      area += points[i][0] * points[j][1] - points[j][0] * points[i][1];
    }

    return area / 2;
  }

  // ---------------------------------------------------------------------------
  // Coefficient spectrum
  // ---------------------------------------------------------------------------

  function updateCoeffReadout(coefficients) {
    if (!coefficients.length) {
      setText(els.coeffIndexText, "Coefficient --");
      setText(els.coeffValueText, "--");
      setText(els.coeffHint, "No Fourier coefficient array was returned for this record.");
      return;
    }

    let index = state.coeffHoverIndex;

    if (index < 0 || index >= coefficients.length) {
      index = maxAbsCoefficientIndex(coefficients);
    }

    const value = coefficients[index];

    setText(els.coeffIndexText, `Coefficient #${index}`);
    setText(els.coeffValueText, Number(value).toFixed(6));
    setText(
      els.coeffHint,
      "Move mouse over bars to inspect coefficient index and value."
    );
  }

  function maxAbsCoefficientIndex(coefficients) {
    let bestIndex = 0;
    let bestValue = -Infinity;

    coefficients.forEach((value, index) => {
      const abs = Math.abs(Number(value));

      if (Number.isFinite(abs) && abs > bestValue) {
        bestValue = abs;
        bestIndex = index;
      }
    });

    return bestIndex;
  }

  function drawCoefficientSpectrum(coefficients) {
    if (els.coeffSpectrumOverlay) {
      els.coeffSpectrumOverlay.classList.toggle("is-visible", !coefficients.length);
    }

    const renderer = ensureCoeffRenderer();

    if (!renderer) return;

    try {
      renderer.drawSpectrum(coefficients, state.coeffHoverIndex);
    } catch (error) {
      addLog(`WebGL coefficient spectrum draw failed: ${error.message}`, "error");
    }
  }

  function handleCoeffMouseMove(event) {
    const record = getSelectedPreviewRecord();

    if (!record) return;

    const coefficients = getRecordCoefficientArray(record);

    if (!coefficients.length || !els.coeffSpectrumCanvas) return;

    const rect = els.coeffSpectrumCanvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const index = clamp(
      Math.floor((x / Math.max(1, rect.width)) * coefficients.length),
      0,
      coefficients.length - 1
    );

    state.coeffHoverIndex = index;
    updateCoeffReadout(coefficients);
    drawCoefficientSpectrum(coefficients);
  }

  function handleCoeffMouseLeave() {
    const record = getSelectedPreviewRecord();
    const coefficients = record ? getRecordCoefficientArray(record) : [];

    state.coeffHoverIndex = -1;
    updateCoeffReadout(coefficients);
    drawCoefficientSpectrum(coefficients);
  }

  // ---------------------------------------------------------------------------
  // Events
  // ---------------------------------------------------------------------------

  function debounce(fn, delay = 150) {
    let timer = 0;

    return (...args) => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => fn(...args), delay);
    };
  }

  function bindEvents() {
    els.selectDbBtn?.addEventListener("click", selectDatabase);
    els.selectFolderBtn?.addEventListener("click", selectImageFolder);
    els.initializeBtn?.addEventListener("click", initializeArchive);
    els.refreshImagesBtn?.addEventListener("click", loadImages);

    els.searchInput?.addEventListener(
      "input",
      debounce(() => {
        renderImages();
      }, 80)
    );

    els.existingOnlyInput?.addEventListener("change", renderImages);

    els.polygonPointsInput?.addEventListener(
      "change",
      debounce(() => {
        reloadCurrentRecords();
      }, 250)
    );

    els.opacityInput?.addEventListener("input", scheduleDraw);
    els.lineWidthInput?.addEventListener("input", scheduleDraw);
    els.fitModeSelect?.addEventListener("change", scheduleDraw);
    els.showLabelsInput?.addEventListener("change", scheduleDraw);
    els.showVerticesInput?.addEventListener("change", scheduleDraw);

    els.canvas?.addEventListener("click", handleCanvasClick);
    els.canvas?.addEventListener("mousemove", handleCanvasMouseMove);
    els.canvas?.addEventListener("mouseleave", handleCanvasMouseLeave);

    els.coeffSpectrumCanvas?.addEventListener("mousemove", handleCoeffMouseMove);
    els.coeffSpectrumCanvas?.addEventListener("mouseleave", handleCoeffMouseLeave);

    window.addEventListener("resize", () => {
      scheduleDraw();
      renderSelectedDefectPreview();
    });

    if (window.ResizeObserver && els.canvas?.parentElement) {
      state.resizeObserver = new ResizeObserver(() => {
        scheduleDraw();
        renderSelectedDefectPreview();
      });
      state.resizeObserver.observe(els.canvas.parentElement);
    }

    document.addEventListener("keydown", (event) => {
      const tagName = event.target?.tagName?.toLowerCase();

      if (["input", "textarea", "select"].includes(tagName)) {
        return;
      }

      if (event.key === "ArrowDown") {
        event.preventDefault();
        selectRelativeImage(1);
      } else if (event.key === "ArrowUp") {
        event.preventDefault();
        selectRelativeImage(-1);
      }
    });
  }

  function syncUrlParams() {
    const params = new URLSearchParams(window.location.search);

    const dbPath =
      params.get("db_path") ||
      params.get("database_path") ||
      params.get("sqlite_path") ||
      "";

    const imageRoot =
      params.get("image_root") ||
      params.get("image_folder") ||
      params.get("image_repository") ||
      "";

    if (dbPath) {
      setValue(els.dbPathInput, dbPath);
    }

    if (imageRoot) {
      setValue(els.imageRootInput, imageRoot);
    }

    const autoinit = params.get("autoinit") || params.get("auto_init");

    if (autoinit === "1" || autoinit === "true") {
      window.setTimeout(initializeArchive, 200);
    }
  }

  function initDefaults() {
    if (els.polygonPointsInput && !els.polygonPointsInput.value) {
      els.polygonPointsInput.value = "256";
    }

    if (els.opacityInput) {
      els.opacityInput.value = "0";
    }

    if (
      els.lineWidthInput &&
      (!els.lineWidthInput.value || String(els.lineWidthInput.value).trim() === "3")
    ) {
      els.lineWidthInput.value = "2.25";
    }

    if (els.fitModeSelect && !els.fitModeSelect.value) {
      els.fitModeSelect.value = "contain";
    }

    if (els.showLabelsInput && els.showLabelsInput.checked === false) {
      els.showLabelsInput.checked = true;
    }
  }

  async function init() {
    collectElements();

    try {
      ensureViewerRenderer();
      ensureCoeffRenderer();
    } catch (error) {
      addLog(`WebGL initialization failed: ${error.message}`, "error");
      toast(`WebGL 初始化失败：${error.message}`, "error", 8000);
      setViewerPlaceholder(`WebGL initialization failed: ${error.message}`, true);
    }

    initDefaults();
    bindEvents();
    syncUrlParams();

    updateSummary({});
    renderImages();
    renderCurrentImagePanel();
    renderRecords();
    scheduleDraw();

    addLog(`Frontend ${VERSION} ready.`);
    setText(els.sessionMessage, "Ready. Select a database and image repository.");

    await checkHealth();
  }

  window.FSDViewer = {
    VERSION,
    state,

    initializeArchive,
    loadImages,
    selectImage,
    reloadCurrentRecords,
    checkHealth,

    renderImages,
    renderRecords,
    renderCurrentImagePanel,
    renderSelectedDefectPreview,
    drawScene
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();