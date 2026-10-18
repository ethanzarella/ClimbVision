"""Browser-side shared playback for two already analyzed H.264 MP4 clips."""
import base64
import html
import streamlit.components.v1 as components


def synchronized_videos(left_video: bytes, right_video: bytes, left_name: str, right_name: str):
    """One control bar commands both video elements in the same HTML document."""
    left_src = base64.b64encode(left_video).decode('ascii')
    right_src = base64.b64encode(right_video).decode('ascii')
    markup = f'''<!doctype html><html><head><meta charset="utf-8"><style>
    body {{margin:0;background:#0e1117;color:#eee;font:14px system-ui, sans-serif}}
    .toolbar {{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:0 0 12px}}
    button {{background:#262d39;border:1px solid #56606f;border-radius:8px;color:white;padding:9px 16px;cursor:pointer}}
    button:hover {{background:#384456}}
    .videos {{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
    .videos video {{display:block;width:100%;max-height:480px;background:black;border-radius:8px}}
    .videos label {{display:block;margin-bottom:7px;overflow-wrap:anywhere}}
    input[type=range] {{flex:1;min-width:100px}}
    .note {{font-size:12px;color:#aaa;margin-top:10px}}
    </style></head><body>
    <div class="toolbar"><button id="play">▶ Play both</button><button id="pause">Ⅱ Pause both</button>
    <button id="restart">↺ Restart both</button><label for="seek">Seek together</label>
    <input id="seek" type="range" min="0" max="100" value="0" step="0.1"></div>
    <div class="videos"><div><label>Attempt A — {html.escape(left_name)}</label><video id="a" playsinline preload="metadata" controls src="data:video/mp4;base64,{left_src}"></video></div>
    <div><label>Attempt B — {html.escape(right_name)}</label><video id="b" playsinline preload="metadata" controls src="data:video/mp4;base64,{right_src}"></video></div></div>
    <div class="note">The two clips start together in video time; different climbing start times and movement sequences are not automatically aligned.</div>
    <script>
    const a=document.getElementById('a'),b=document.getElementById('b'),seek=document.getElementById('seek');
    const videos=[a,b];
    document.getElementById('play').onclick=async()=>{{
      videos.forEach(v=>{{v.pause();v.currentTime=0;}});
      await Promise.allSettled(videos.map(v=>v.play()));
    }};
    document.getElementById('pause').onclick=()=>videos.forEach(v=>v.pause());
    document.getElementById('restart').onclick=()=>{{videos.forEach(v=>{{v.pause();v.currentTime=0;}});seek.value=0;}};
    seek.oninput=()=>{{const portion=Number(seek.value)/100;videos.forEach(v=>{{if(Number.isFinite(v.duration)&&v.duration>0)v.currentTime=v.duration*portion;}});}};
    a.ontimeupdate=()=>{{if(Number.isFinite(a.duration)&&a.duration>0)seek.value=100*a.currentTime/a.duration;}};
    </script></body></html>'''
    components.html(markup, height=660, scrolling=False)
