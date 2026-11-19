// All symbols and specimen diagrams are drawn here. No external image or font requests.
const paths = {
  cube: '<path d="m12 2 8 4.5v10L12 21l-8-4.5v-10L12 2Zm-8 4.5 8 4.7 8-4.7M12 11.2V21"/>',
  layers: '<path d="m12 3 9 5-9 5-9-5 9-5Zm-8 9 8 4.5 8-4.5m-16 5 8 4.5 8-4.5"/>',
  clock: '<circle cx="12" cy="12" r="8"/><path d="M12 7.5V12l3 2"/>',
  book: '<path d="M3.5 5c3.4-.9 5.8-.5 8.5 1.5C14.8 4.6 17.3 4.2 20.5 5v13c-3.3-.8-5.8-.3-8.5 1.6-2.7-1.9-5.2-2.4-8.5-1.6V5ZM12 7v12"/>',
  arrow: '<path d="M4 12h15m-6-6 6 6-6 6"/>',
  chevron: '<path d="m9 6 6 6-6 6"/>',
  down: '<path d="m6 9 6 6 6-6"/>',
  check: '<path d="m5 12 4.3 4.2L19 6.8"/>',
  play: '<path d="m8 4.8 11.2 7.2L8 19.2V4.8Z"/>',
  mouse: '<rect x="6.3" y="2.7" width="11.4" height="18.6" rx="5.7"/><path d="M12 6v4"/>',
  rotate: '<path d="M3.5 7v5h5M4 11a8 8 0 1 1 2.2 6.5"/>',
  orbit: '<ellipse cx="12" cy="12.5" rx="9.5" ry="4.5" transform="rotate(-24 12 12.5)"/><circle cx="12" cy="12" r="2.2"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  minus: '<path d="M5 12h14"/>',
  history: '<path d="M3 4.5v5h5M3.8 9a8.5 8.5 0 1 1 .6 7M12 7.5v5l3.2 1.8"/>',
  shield: '<path d="m12 2.5 7.8 3.1v6c0 4.9-3.1 7.8-7.8 10-4.7-2.2-7.8-5.1-7.8-10v-6L12 2.5Z"/><path d="m8.4 12 2.5 2.5 4.7-5"/>',
  disk: '<rect x="3.1" y="4" width="17.8" height="13" rx="2"/><path d="M9 20h6m-3-3v3"/>',
  close: '<path d="m6 6 12 12M18 6 6 18"/>',
  file: '<path d="M5.5 3.5h8l5 5V21h-13V3.5Z"/><path d="M13 4v5h5m-9 4h6m-6 3h5"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 10v6m0-9.1v.2"/>',
  target: '<circle cx="12" cy="12" r="6.7"/><path d="M12 2v4m0 12v4M2 12h4m12 0h4"/><circle cx="12" cy="12" r="1.3"/>',
  list: '<rect x="4" y="3" width="16" height="18" rx="2"/><path d="M8 8h1m3 0h4m-8 4h1m3 0h4m-8 4h6"/>',
  pen: '<path d="M4 20l4.6-1L20 7.7 16.3 4 5 15.3 4 20Zm9.6-13.2 3.6 3.6"/>',
  lock: '<rect x="5" y="10" width="14" height="11" rx="2"/><path d="M8 10V7a4 4 0 0 1 8 0v3m-4 4v3"/>',
  ticket: '<path d="M3 6h18v4a2 2 0 1 0 0 4v4H3v-4a2 2 0 1 0 0-4V6Z"/><path d="M15 7.5v1.3m0 2.6v1.3m0 2.6v1.3"/>',
  camera: '<path d="M3 7h4l2-3h6l2 3h4v12H3V7Z"/><circle cx="12" cy="13" r="3.5"/>',
  enlarge: '<path d="M4 9V4h5m6 0h5v5m0 6v5h-5m-6 0H4v-5"/>',
};
export const icon = (name, size = 18) => '<svg aria-hidden="true" width="' + size + '" height="' + size + '" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.65" stroke-linecap="round" stroke-linejoin="round">' + paths[name] + '</svg>';
export const logo = '<svg aria-hidden="true" viewBox="0 0 40 40" fill="none" width="37" height="37"><rect width="40" height="40" rx="11" fill="#095e55"/><path d="m20 6.5 12.2 7-5.1 3-7.1-4.1-7.3 4.2v8.5l-5.1 3V13.6L20 6.5Z" fill="#e0f1da"/><path d="m15.4 18.1 4.6-2.7 4.6 2.7v5.4L20 26.2l-4.6-2.7v-5.4Z" fill="#95d8aa"/><path d="m32.2 16.7.1 9.9L20 33.7l-9.5-5.5 5.1-3 4.4 2.6 7.1-4.1v-4.1l5.1-2.9Z" fill="white"/></svg>';

function rounded(ctx, x, y, w, h, radius, fill) {
  ctx.fillStyle = fill;
  ctx.beginPath(); ctx.roundRect(x, y, w, h, radius); ctx.fill();
}
function linear(ctx, a, b, stops) {
  const g = ctx.createLinearGradient(...a, ...b);
  stops.forEach(([position, color]) => g.addColorStop(position, color));
  return g;
}

// Normalized drafting surface, with the identical geometry used for the source and the map.
export function drawSpecimen(canvas, id, { map = false, thumbnail = false } = {}) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  const h = 176, w = canvas.width * h / canvas.height;
  ctx.setTransform(canvas.height / h, 0, 0, canvas.height / h, 0, 0);
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = map ? '#e2e9e3' : thumbnail ? '#e8eeea' : '#eef1ec';
  ctx.fillRect(0, 0, w, h);
  if (!thumbnail) {
    ctx.fillStyle = '#c5d0c6';
    for (let x=6; x<w; x+=12) for (let y=5; y<h; y+=12) ctx.fillRect(x,y,1,1);
    ctx.strokeStyle = '#a6b8af'; ctx.lineWidth = 1.25;
    for (const x of [11, w-11]) for (const y of [11,h-11]) {
      ctx.beginPath(); ctx.moveTo(x, y + (y<20 ? 6 : -6)); ctx.lineTo(x,y); ctx.lineTo(x + (x<20 ? 6 : -6),y); ctx.stroke();
    }
  }
  ctx.save();
  ctx.translate(w/2, h/2);
  if (thumbnail) { ctx.scale(.98,.98); ctx.rotate(id === '03' ? -.12 : -.20); }
  const shadow = ctx.createRadialGradient(1,5,4,1,5,69);
  shadow.addColorStop(0,'#657b6e3d'); shadow.addColorStop(1,'#657b6e00');
  ctx.fillStyle=shadow; ctx.fillRect(-72,-65,144,145);
  // Bent, straight and occluded leads share the same stamped metal profile.
  [-25,0,25].forEach((x, i) => {
    const points = id === '01' && i === 0 ? [[x,12],[x,35],[x-2,43],[x-16,54],[x-21,66]] : [[x,12],[x,37],[x,47],[x,67]];
    ctx.lineCap='butt'; ctx.lineJoin='round';
    for (const [offset,width,color] of [[1.3,6.5,'#758680'],[0,5.4,'#aebbb4'],[-1.15,1.6,'#e0e5df']]) {
      ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();
      points.forEach(([px,py],j)=>j ? ctx.lineTo(px+offset,py) : ctx.moveTo(px+offset,py));ctx.stroke();
    }
    rounded(ctx,x-4.5,10,9,23,1.5,'#87968c');
    rounded(ctx,x-3.2,10,2,22,1,'#d3dbd0');
  });
  const metal=linear(ctx,[-36,-62],[36,-15],[[0,'#e4e9e1'],[.26,'#a9b7af'],[.56,'#d6dfd4'],[1,'#788c83']]);
  rounded(ctx,-35,-66,70,54,5,'#768a80');
  rounded(ctx,-34,-65,68,51,4,metal);
  ctx.fillStyle='#71857a';ctx.beginPath();ctx.arc(0,-51,8,0,Math.PI*2);ctx.fill();
  ctx.fillStyle=thumbnail?'#e8eeea':'#e9eee8';ctx.beginPath();ctx.arc(0,-50,5.9,0,Math.PI*2);ctx.fill();
  rounded(ctx,-41,-35,82,57,5,'#26302d');
  rounded(ctx,-39.9,-34.3,79.8,53.5,4.4,linear(ctx,[-40,-32],[32,21],[[0,'#515c56'],[.16,'#34413c'],[.7,'#29352f'],[1,'#424e46']]));
  rounded(ctx,-37,-32,74,1.6,.7,'#637169');
  if (!thumbnail) {
    ctx.fillStyle='#c6d0c2';ctx.font='bold 8px monospace';ctx.fillText('D–OPS',-26,-14);
    ctx.fillStyle='#a6b5ab';ctx.font='6.3px monospace';ctx.fillText('TR / 0'+Number(id),-26,-3);
    ctx.fillStyle='#59665c';ctx.beginPath();ctx.arc(27,9,3.4,0,Math.PI*2);ctx.fill();
  }
  if (id==='02') {
    const reflection=linear(ctx,[12,0],[34,0],[[0,'#d6e0d000'],[.57,'#d6e0d05a'],[1,'#d6e0d000']]);
    rounded(ctx,12,-30,22,44,2,reflection);
  }
  if (id==='03') {
    ctx.save();ctx.translate(-22,52);ctx.rotate(-.15);
    rounded(ctx,-33,-15,68,36,2,'#8a9b96');rounded(ctx,-33,-17,68,35,2,'#a9b6ad');
    ctx.beginPath();ctx.rect(-32,-17,66,34);ctx.clip();
    ctx.lineWidth=2;ctx.strokeStyle='#8f9f95';
    for(let x=-55;x<45;x+=8){ctx.beginPath();ctx.moveTo(x,19);ctx.lineTo(x+18,-19);ctx.stroke();}
    ctx.restore();
  }
  if (map) {
    const [x,y,r] = id==='01' ? [6,-33,35] : id==='02' ? [23,-9,31] : [-18,36,23];
    const glow=ctx.createRadialGradient(x,y,1,x,y,r);
    const colors=id==='03' ? ['#c5c653ba','#9ccd69a8','#6aa99566','#639e9600'] : ['#df5227fa','#f29b37ed','#ebcf52a8','#8dc58b00'];
    [0,.3,.63,1].forEach((v,i)=>glow.addColorStop(v,colors[i]));ctx.fillStyle=glow;ctx.fillRect(x-r,y-r,r*2,r*2);
  }
  ctx.restore();
}
