const $ = id => document.getElementById(id);
const ns = 'http://www.w3.org/2000/svg';
let selectedFile = null, busy = false, previewUrl = null, activeRegion = null;

function message(text, error=false) {
  $('message').textContent = text;
  $('message').hidden = !text;
  $('message').classList.toggle('error', error);
}
function resetResult() {
  for (const id of ['heat-value','sheet-value','year-value','time-value']) $(id).textContent='—';
  $('result-status').textContent='Ожидание'; $('result-status').className='status neutral';
  $('heat-status').textContent='После распознавания';
  $('sheet-status').textContent='После распознавания'; $('full-text').textContent='Результат ещё не получен.';
  $('comparison').replaceChildren(); const empty=document.createElement('div'); empty.className='comparison-empty';
  empty.textContent='Номера будут сопоставлены после распознавания.'; $('comparison').append(empty);
  $('overlay').replaceChildren(); $('crops').replaceChildren(); $('crop-count').textContent='0';
  $('crop-empty').hidden=false; $('download-csv').hidden=true; $('download-json').hidden=true;
  $('rotation-info').textContent=''; activeRegion=null;
}
function choose(file) {
  if (busy || !file) return;
  if (!/\.(jpe?g|png)$/i.test(file.name)) { message('Выберите фотографию JPG или PNG.', true); return; }
  if (file.size>40*1024*1024 || !file.size) { message('Выберите непустой файл размером до 40 МБ.', true); return; }
  selectedFile=file; resetResult();
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl=URL.createObjectURL(file); $('photo').src=previewUrl;
  $('image-empty').hidden=true; $('image-wrap').hidden=false; $('file-meta').hidden=false;
  $('file-name').textContent=file.name; $('file-size').textContent=(file.size/1024/1024).toFixed(2)+' МБ';
  $('recognize').disabled=false; $('image-info').textContent='Предварительный просмотр';
  message('Фото выбрано. Нажмите «Распознать маркировку».');
}
$('file').addEventListener('change', event => choose(event.target.files[0]));
$('dropzone').addEventListener('keydown',event=>{ if (event.key==='Enter'||event.key===' ') { event.preventDefault(); if(!busy) $('file').click(); } });
for (const name of ['dragenter','dragover']) $('dropzone').addEventListener(name,event=>{event.preventDefault();if(!busy)$('dropzone').classList.add('dragover');});
for (const name of ['dragleave','drop']) $('dropzone').addEventListener(name,event=>{event.preventDefault();$('dropzone').classList.remove('dragover');});
$('dropzone').addEventListener('drop',event=>choose(event.dataTransfer.files[0]));
$('show-boxes').addEventListener('change',()=>{$('overlay').toggleAttribute('hidden',!$('show-boxes').checked);});

function selectRegion(index, scroll=false) {
  activeRegion=index;
  for (const item of document.querySelectorAll('[data-region]')) item.classList.toggle('active',Number(item.dataset.region)===index);
  if (scroll) document.getElementById('crop-'+index)?.scrollIntoView({behavior:'smooth',block:'nearest'});
}
function showRegions(data) {
  $('overlay').setAttribute('viewBox',`0 0 ${data.image_width} ${data.image_height}`);
  $('overlay').setAttribute('preserveAspectRatio','xMidYMid meet');
  $('overlay').replaceChildren(); $('crops').replaceChildren();
  const regions=data.regions||[];
  $('crop-count').textContent=regions.length; $('crop-empty').hidden=regions.length>0;
  for(const region of regions) {
    const poly=document.createElementNS(ns,'polygon');
    poly.setAttribute('points',region.points.map(p=>p.join(',')).join(' '));
    poly.dataset.region=region.index; poly.setAttribute('tabindex','0'); poly.setAttribute('role','button');
    poly.setAttribute('aria-label','Область '+region.index+': '+region.text);
    poly.addEventListener('click',()=>selectRegion(region.index,true));
    poly.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();selectRegion(region.index,true);}});
    $('overlay').append(poly);
    const label=document.createElementNS(ns,'text'); label.setAttribute('x',region.points[0][0]+4);label.setAttribute('y',region.points[0][1]+22);label.textContent=region.index;$('overlay').append(label);
    const card=document.createElement('button');card.type='button';card.className='crop';card.id='crop-'+region.index;card.dataset.region=region.index;
    const caption=document.createElement('div');caption.className='crop-caption';
    const number=document.createElement('strong');number.textContent='ОБЛАСТЬ '+String(region.index).padStart(2,'0');
    const confidence=document.createElement('span');confidence.textContent=Math.round(region.confidence*100)+'% уверенность';caption.append(number,confidence);
    const holder=document.createElement('div');holder.className='crop-image';
    const image=document.createElement('img');image.src=region.crop_url;image.alt='Вырезка строки '+region.index;holder.append(image);
    const text=document.createElement('div');text.className='crop-text';text.textContent=region.text;
    card.append(caption,holder,text);card.addEventListener('click',()=>selectRegion(region.index));$('crops').append(card);
  }
}
function showComparison(comparison) {
  const labels={heat_id:'Плавка',sheet_id:'Лист',heat_year:'Год'};
  const states={match:'✓ Совпадает',mismatch:'≠ Отличается',unavailable:'Нет в имени',not_found:'Нет в OCR'};
  $('comparison').replaceChildren();
  for (const key of ['heat_id','sheet_id','heat_year']) {
    const field=comparison.fields[key], row=document.createElement('div');row.className='compare-row';row.dataset.status=field.status;
    const top=document.createElement('div');top.className='compare-top';
    const title=document.createElement('span');title.textContent=labels[key];const status=document.createElement('span');status.className='compare-status';status.textContent=states[field.status];top.append(title,status);
    const values=document.createElement('div');values.className='compare-values';
    for (const [label,value] of [['OCR',field.ocr],['Имя файла',field.filename]]) {
      const cell=document.createElement('div'), name=document.createElement('span'), code=document.createElement('strong');
      name.textContent=label;code.textContent=value==null||value===''?'—':value;cell.append(name,code);values.append(cell);
    }
    row.append(top,values);$('comparison').append(row);
  }
}
function showResult(data) {
  $('photo').src=data.image_url;
  $('image-info').textContent=`${data.image_width} × ${data.image_height} px · ${(data.regions||[]).length} текстовых областей`;
  $('rotation-info').textContent='Поворот OCR: '+data.rotation+'°';
  $('heat-value').textContent=data.heat_id||'Не найден';$('sheet-value').textContent=data.sheet_id||'Не найден';
  $('heat-status').textContent=data.heat_id?'Распознан на фотографии':(data.heat_ambiguous?'Несколько противоречивых номеров':'Номер не найден');
  $('year-value').textContent=data.heat_year||'Не найден';$('time-value').textContent=data.elapsed_seconds.toFixed(2)+' с';
  const sheetStates={ok:'Распознан на фотографии',low_confidence:'Проверьте: низкая уверенность',ambiguous:'Несколько противоречивых номеров',not_found:'Не прочитан или формат не поддержан'};
  $('sheet-status').textContent=sheetStates[data.sheet_status]||'Не найден';
  const status=$('result-status');status.textContent=data.status==='ok'?'Распознано':data.status==='not_found'?'Текст не найден':'Нужна проверка';
  status.className='status '+(data.status==='ok'?'success':'warning');
  $('full-text').textContent=data.text||'Текст не найден';
  showRegions(data);showComparison(data.comparison);
  $('download-csv').href=data.csv_url;$('download-csv').hidden=false;
  $('download-json').href=data.json_url;$('download-json').hidden=false;
}
$('recognize').addEventListener('click',async()=>{
  if(busy||!selectedFile)return;
  busy=true;$('recognize').disabled=true;$('file').disabled=true;$('spinner').hidden=false;$('button-arrow').hidden=true;
  $('button-label').textContent='Распознаём…';resetResult();$('result-status').textContent='Обработка';
  message('Обрабатываем фотографию. Первый запуск загружает модели в память; на CPU это может занять несколько минут.');
  try {
    const response=await fetch('/api/recognize',{method:'POST',headers:{'Content-Type':'application/octet-stream','X-Metal-Upload':'1','X-File-Name':encodeURIComponent(selectedFile.name)},body:selectedFile});
    const data=await response.json();
    if(!response.ok)throw Error(data.error||'Ошибка распознавания. Проверьте терминал сервера.');
    showResult(data);message('Распознавание завершено. Проверьте номера и вырезки; CSV содержит только пять выбранных колонок.');
  } catch(error) {
    $('result-status').textContent='Ошибка';$('result-status').className='status warning';message(error.message,true);
  } finally {
    busy=false;$('recognize').disabled=!selectedFile;$('file').disabled=false;$('spinner').hidden=true;$('button-arrow').hidden=false;$('button-label').textContent='Распознать маркировку';
  }
});
