const form = document.getElementById('detectForm');
const inputPreview = document.getElementById('inputPreview');
const outputPreview = document.getElementById('outputPreview');
const totalPests = document.getElementById('totalPests');
const countList = document.getElementById('countList');
const reportLinks = document.getElementById('reportLinks');
const summaryLink = document.getElementById('summaryLink');
const detailLink = document.getElementById('detailLink');
const jsonLink = document.getElementById('jsonLink');

const refreshHistoryBtn = document.getElementById('refreshHistoryBtn');
const historyList = document.getElementById('historyList');
const methodForm = document.getElementById('methodForm');
const pestName = document.getElementById('pestName');
const methodText = document.getElementById('methodText');
const methodList = document.getElementById('methodList');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(form);

  const localFile = document.getElementById('imageInput').files[0];
  if (localFile) inputPreview.src = URL.createObjectURL(localFile);

  totalPests.textContent = '总数：检测中...';
  countList.innerHTML = '检测中，请稍候...';
  reportLinks.classList.add('hidden');

  const response = await fetch('/api/detect', { method: 'POST', body: formData });
  const data = await response.json();

  if (!response.ok) {
    totalPests.textContent = '总数：0';
    countList.innerHTML = `检测失败：${data.error || '未知错误'}`;
    return;
  }

  outputPreview.src = data.annotated_image;
  totalPests.textContent = `总数：${data.total_pests}`;

  const entries = Object.entries(data.counts);
  countList.innerHTML = entries.length
    ? entries.map(([cls, count]) => `<span class="badge">${cls}: ${count}</span>`).join('')
    : '未检测到害虫目标';

  summaryLink.href = data.report_files.summary_csv;
  detailLink.href = data.report_files.detail_csv;
  jsonLink.href = data.report_files.json;
  reportLinks.classList.remove('hidden');

  await loadHistory();
});

async function loadHistory() {
  const resp = await fetch('/api/history');
  const data = await resp.json();

  if (!Array.isArray(data) || data.length === 0) {
    historyList.innerHTML = '暂无历史记录';
    return;
  }

  historyList.innerHTML = data
    .map(
      (item) => `
      <div class="history-item">
        <div>
          <strong>#${item.id}</strong> ${item.created_at} ｜ 总数：${item.total_pests}
          <div>${Object.entries(item.counts).map(([k, v]) => `${k}:${v}`).join('，') || '无检测目标'}</div>
        </div>
        <div class="history-actions">
          <a class="mini-link" href="${item.annotated_image}" target="_blank">查看结果图</a>
          <a class="mini-link" href="${item.report_files.summary_csv}">汇总</a>
          <a class="mini-link" href="${item.report_files.detail_csv}">明细</a>
          <button class="danger" onclick="deleteHistory(${item.id})">删除</button>
        </div>
      </div>
    `,
    )
    .join('');
}

async function deleteHistory(id) {
  const resp = await fetch(`/api/history/${id}`, { method: 'DELETE' });
  if (resp.ok) await loadHistory();
}

async function loadMethods() {
  const resp = await fetch('/api/control-methods');
  const data = await resp.json();

  if (!Array.isArray(data) || !data.length) {
    methodList.innerHTML = '暂无防治方法';
    return;
  }

  methodList.innerHTML = data
    .map(
      (m) => `
        <div class="method-item">
          <h4>${m.pest_name}</h4>
          <p>${m.method_text}</p>
          <small>更新时间：${m.updated_at}</small>
        </div>
      `,
    )
    .join('');
}

methodForm.addEventListener('submit', async (e) => {
  e.preventDefault();

  const resp = await fetch('/api/control-methods', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ pest_name: pestName.value.trim(), method_text: methodText.value.trim() }),
  });

  if (resp.ok) {
    methodForm.reset();
    await loadMethods();
  }
});

refreshHistoryBtn.addEventListener('click', loadHistory);
loadHistory();
loadMethods();


