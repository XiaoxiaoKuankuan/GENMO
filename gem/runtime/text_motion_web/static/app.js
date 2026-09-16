/* GENMO 网页交互入口：读取本地模型和任务，生成成功后手动播放视频。
   所有模型路径和提示文本都作为纯文本显示，避免将输入解释为 HTML。 */
"use strict";
const $ = (id) => document.getElementById(id);
const state = {models: [], jobs: [], activeId: null, selectedId: null, pending: false, lastSignature: "", lastStatusKey: "", connected: false, canRegister: true};
const stageNames = {queued: "等待开始", loading: "正在加载模型", generating: "正在生成动作", rendering: "正在渲染并检查视频", done: "生成完成 · 点击播放查看", failed: "生成失败 · 可以重新尝试"};

async function api(path, payload) {
  const response = await fetch(path, payload === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(payload)});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `请求失败 (${response.status})`);
  return data;
}
function showError(message) { $("error").textContent = message; $("error").hidden = !message; }
function duration() { $("duration").textContent = `${Number(($("num-frames").value / 30).toFixed(2))} 秒 · 30 FPS`; }
function busy() {
  const generating = state.pending || Boolean(state.activeId);
  $("generate").disabled = generating || !state.models.length || !state.connected;
  $("generate").firstElementChild.textContent = generating ? "生成进行中…" : "生成动作视频";
}
function modelDetail() {
  const model = state.models.find((m) => m.id === $("model").value);
  $("model-detail").textContent = model ? `${model.contract.max_text_len} token · ${model.path || model.name}` : "未发现可用模型，请联系站点维护者。";
}
async function loadModels(selectedId) {
  const data = await api("/api/models");
  const previous = selectedId || $("model").value;
  state.models = data.models;
  state.canRegister = data.capabilities?.register_models !== false;
  if ($("add-model")) $("add-model").hidden = !state.canRegister;
  const options = data.models.map((model) => {
    const option = document.createElement("option");
    option.value = model.id;
    option.textContent = `${model.is_default ? "默认 · " : ""}${model.name} · step ${model.global_step ?? "未知"}`;
    return option;
  });
  if (!options.length) {
    const option = document.createElement("option"); option.value = "";
    option.textContent = data.scanning ? "正在发现本地文本模型…" : "请添加文本模型";
    options.push(option);
  }
  $("model").replaceChildren(...options);
  if (data.models.some((m) => m.id === previous)) $("model").value = previous;
  modelDetail(); busy();
  return data.scanning;
}
function selectVideo(job) {
  if (job.status !== "done") return;
  const video = $("video");
  video.pause();
  video.src = job.video_url;
  video.poster = job.thumbnail_url;
  video.load();
  video.hidden = false;
  $("empty-preview").hidden = true;
  $("video-title").textContent = job.prompt;
  $("video-meta").textContent = `${job.model.name} · ${job.num_frames} 帧 / ${Number((job.num_frames / 30).toFixed(2))} 秒 · DDIM ${job.ddim_steps}`;
  $("download").href = job.video_url;
  $("download").hidden = false;
  state.selectedId = job.id;
  document.querySelectorAll(".history-card").forEach((card) => card.classList.toggle("selected", card.dataset.id === job.id));
}
function textNode(tag, text, className) {
  const node = document.createElement(tag); node.textContent = text;
  if (className) node.className = className;
  return node;
}
async function reuse(job) {
  try {
    if (!state.models.some((m) => m.id === job.model_id)) {
      if (!state.canRegister || !job.model.path) {
        await loadModels();
        if (!state.models.some((m) => m.id === job.model_id)) throw new Error("此历史使用的模型暂不可用，请选择列表中的模型。");
      } else {
        const data = await api("/api/models", {path: job.model.path});
        await loadModels(data.model.id);
      }
    }
    $("model").value = job.model_id;
    $("prompt").value = job.prompt;
    $("num-frames").value = job.num_frames;
    $("ddim-steps").value = job.ddim_steps;
    modelDetail(); duration();
    $("prompt").focus();
  } catch (error) { showError(error.message); }
}
function renderHistory() {
  $("history-count").textContent = state.jobs.length;
  if (!state.jobs.length) {
    $("history-list").replaceChildren(textNode("p", "还没有生成记录。你的动作视频会保存在这里。", "history-empty"));
    return;
  }
  const cards = state.jobs.map((job) => {
    const card = document.createElement("article");
    card.className = `history-card${job.id === state.selectedId ? " selected" : ""}`;
    card.dataset.id = job.id;
    if (job.status === "done") {
      const img = document.createElement("img"); img.src = job.thumbnail_url;
      img.alt = `动作缩略图：${job.prompt}`; img.loading = "lazy";
      card.append(img);
    } else card.append(textNode("div", stageNames[job.status], job.status === "failed" ? "failed-label" : "hint"));
    const body = document.createElement("div"); body.className = "card-body";
    body.append(textNode("h3", job.prompt), textNode("p", job.model.name),
      textNode("p", `${job.num_frames} 帧 · 30 FPS · DDIM ${job.ddim_steps} · ${new Date(job.created_at).toLocaleString("zh-CN")}`));
    if (job.status === "failed") body.append(textNode("p", job.error));
    const actions = document.createElement("div"); actions.className = "actions";
    if (job.status === "done") {
      const play = textNode("button", "回看视频"); play.type = "button";
      play.addEventListener("click", () => { selectVideo(job); $("video").scrollIntoView({block: "center", behavior: "smooth"}); });
      actions.append(play);
    }
    const use = textNode("button", "复用参数"); use.type = "button";
    use.addEventListener("click", () => reuse(job)); actions.append(use);
    body.append(actions); card.append(body); return card;
  });
  $("history-list").replaceChildren(...cards);
}
function renderStatus(job) {
  if (!job) return;
  $("status-text").textContent = stageNames[job.status];
  $("elapsed").textContent = `耗时 ${job.elapsed_seconds.toFixed(1)} 秒`;
  const stages = ["loading", "generating", "rendering", "done"];
  const index = stages.indexOf(job.status === "failed" ? job.failed_stage : job.status);
  document.querySelectorAll(".steps li").forEach((li, i) => li.classList.toggle("active", i <= index));
  const statusKey = `${job.id}:${job.status}`;
  if (statusKey !== state.lastStatusKey) {
    showError(job.status === "failed" ? `${stageNames[job.failed_stage] || job.failed_stage}失败\n${job.error}` : "");
    state.lastStatusKey = statusKey;
  }
}
async function refreshHistory() {
  const data = await api("/api/history");
  const previousActive = state.activeId;
  if (!state.connected) { showError(""); state.lastStatusKey = ""; }
  state.jobs = data.jobs; state.activeId = data.active_id; state.connected = true;
  const completed = data.jobs.find((job) => job.id === previousActive && job.status === "done");
  if (completed) selectVideo(completed);
  if (!state.selectedId) {
    const last = data.jobs.find((job) => job.status === "done");
    if (last) selectVideo(last);
  }
  const signature = JSON.stringify(data.jobs.map((job) => [job.id, job.status]));
  if (signature !== state.lastSignature) { state.lastSignature = signature; renderHistory(); }
  renderStatus(data.jobs.find((job) => job.id === data.active_id) || data.jobs[0]);
  if (data.recovery_errors.length) showError(`部分历史记录读取失败：${data.recovery_errors.join("\n")}`);
  busy();
}
$("num-frames").addEventListener("input", duration);
$("model").addEventListener("change", modelDetail);
$("register")?.addEventListener("click", async () => {
  $("register").disabled = true; $("register-status").textContent = "正在校验 checkpoint 内容…";
  try {
    const data = await api("/api/models", {path: $("model-path").value});
    await loadModels(data.model.id);
    $("register-status").textContent = "校验通过，已选择此模型。";
  } catch (error) { $("register-status").textContent = error.message; }
  finally { $("register").disabled = false; }
});
$("generate-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.pending || state.activeId) return;
  const prompt = $("prompt").value;
  if (!prompt.trim()) { showError("请输入动作描述，不能只有空白。"); return; }
  state.pending = true; busy(); showError("");
  try {
    const job = await api("/api/jobs", {model_id: $("model").value, prompt,
      num_frames: Number($("num-frames").value), ddim_steps: Number($("ddim-steps").value)});
    state.activeId = job.id; renderStatus(job);
    await refreshHistory();
  } catch (error) { showError(error.message); }
  finally { state.pending = false; busy(); }
});
$("video").addEventListener("error", () => showError("浏览器未能读取视频，请从历史重新选择；详细动作和视频仍保存在本地任务目录。"));
let scanning = true;
async function poll() {
  try {
    if (scanning || !state.models.length) scanning = await loadModels();
    await refreshHistory();
  } catch (error) { state.connected = false; busy(); showError(`服务暂时无法连接：${error.message}。页面会自动重试。`); }
  finally { setTimeout(poll, 1200); }
}
duration(); poll();
