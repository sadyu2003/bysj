const form = document.getElementById('detectForm');
const inputPreview = document.getElementById('inputPreview');
const outputPreview = document.getElementById('outputPreview');
const countList = document.getElementById('countList');
const reportLink = document.getElementById('reportLink');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(form);

  const localFile = document.getElementById('imageInput').files[0];
  if (localFile) {
    inputPreview.src = URL.createObjectURL(localFile);
  }

  countList.innerHTML = '检测中，请稍候...';
  reportLink.classList.add('hidden');

  const response = await fetch('/api/detect', {
    method: 'POST',
    body: formData,
  });

  const data = await response.json();
  if (!response.ok) {
    countList.innerHTML = `检测失败：${data.error || '未知错误'}`;
    return;
  }

  outputPreview.src = data.annotated_image;
  const entries = Object.entries(data.counts);
  if (!entries.length) {
    countList.innerHTML = '未检测到害虫目标';
  } else {
    countList.innerHTML = entries
      .map(([cls, count]) => `<span class="badge">${cls}: ${count}</span>`)
      .join('');
  }

  reportLink.href = data.report;
  reportLink.classList.remove('hidden');
});
