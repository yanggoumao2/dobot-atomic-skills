"""Create a local-only pairing card and a reusable IP-specific TLS certificate."""
import base64
import hashlib
import html
import os
import shutil
import ssl
import subprocess
import tempfile
from pathlib import Path


def prepare_tls(directory, address):
    directory.mkdir(parents=True, exist_ok=True)
    cert, key = (directory / f"{address}.{suffix}" for suffix in ("crt", "key"))
    openssl = shutil.which("openssl") or os.environ.get("NOVA2_OPENSSL")
    if cert.exists() and key.exists():
        try:
            if openssl:
                subprocess.run([openssl, "x509", "-checkend", "86400", "-noout", "-in", str(cert)],
                               check=True, capture_output=True)
            ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER).load_cert_chain(cert, key)
            return cert, key
        except (ssl.SSLError, subprocess.CalledProcessError):
            pass
    if not openssl:
        raise RuntimeError("生成局域网证书需要 openssl.exe，请设置 NOVA2_OPENSSL 后重新启动")
    # Generate privately in a temporary directory. The service only serves its constant HTML.
    with tempfile.TemporaryDirectory(dir=directory) as temp:
        paths = [Path(temp) / p.name for p in (cert, key)]
        subprocess.run([openssl, "req", "-x509", "-newkey", "rsa:2048", "-nodes",
                        "-days", "365", "-subj", "/CN=Nova2 LAN",
                        "-addext", f"subjectAltName=IP:{address}",
                        "-addext", "extendedKeyUsage=serverAuth",
                        "-keyout", str(paths[1]), "-out", str(paths[0])],
                       check=True, capture_output=True)
        for src, dst in zip(paths, (cert, key)):
            os.replace(src, dst)
    return cert, key


def qr_image(url):
    try:
        import cv2
        matrix = cv2.QRCodeEncoder_create().encode(url)
        matrix = cv2.copyMakeBorder(matrix, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=255)
        matrix = cv2.resize(matrix, None, fx=8, fy=8, interpolation=cv2.INTER_NEAREST)
        ok, data = cv2.imencode(".png", matrix)
        if ok:
            return '<img alt="手机扫描打开遥控页面" width="210" src="data:image/png;base64,' + base64.b64encode(data).decode() + '">'
    except (ImportError, AttributeError):
        pass
    return ""


def write_card(directory, url, code, cert, live):
    digest = hashlib.sha256(ssl.PEM_cert_to_DER_cert(cert.read_text())).hexdigest().upper()
    fingerprint = " ".join(digest[i:i + 4] for i in range(0, len(digest), 4))
    mode = "真机" if live else "模拟"
    card = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
    <meta name="viewport" content="width=device-width,initial-scale=1"><title>Nova2 手机配对卡</title>
    <style>body{{background:#eff4f3;color:#182d31;font:17px/1.6 system-ui,sans-serif;margin:5vh auto;padding:20px;max-width:540px}}
    main{{background:white;padding:28px;border-radius:20px}}h1{{margin:0}}code,strong{{display:block;overflow-wrap:anywhere}}
    strong{{font: bold 40px monospace;letter-spacing:4px}}small{{display:block;margin-top:20px;color:#456}}</style>
    <main><h1>Nova2 手机配对</h1><p>{mode}模式 · 手机与电脑连同一 Wi-Fi 后扫码，或在手机浏览器输入以下地址。</p>
    {qr_image(url)}<code>{html.escape(url)}</code><p>本次配对码</p><strong>{code}</strong>
    <small>此服务使用本机签发的 HTTPS 证书。首次访问时核对地址和证书 SHA-256 指纹，再继续访问。<code>{fingerprint}</code></small>
    <small>在手机上选择左臂或右臂。若键盘点动正在占用设备，请先在原窗口按 q 正常退出。</small>
    <small>摇杆满幅速度上限为 12 mm/s。活动范围以启用位置为中心，水平半径 100 mm，上下各 60 mm。请看着机械臂操作，并保持实体急停可及。</small></main></html>"""
    path = directory / "pairing.html"
    temp = path.with_suffix(".tmp")
    temp.write_text(card, encoding="utf-8")
    os.replace(temp, path)
