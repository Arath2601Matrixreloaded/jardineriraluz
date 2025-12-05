import streamlit as st
import sqlite3
import os
import cv2
import numpy as np
import time
import random
import base64
import requests
from datetime import datetime

DB_PATH = "sistema.db"
CAPTURAS_DIR = "capturas"
FOTOS_ROSTRO_DIR = "fotos_rostro"

# --------------------------
# Inicialización de BD
# --------------------------
def get_conn():
    dsn = None
    try:
        sec = st.secrets
        dsn = sec.get("postgres_url")
        if not dsn:
            pg = sec.get("postgres")
            if pg:
                host = pg.get("host")
                db = pg.get("dbname") or pg.get("db") or pg.get("database")
                user = pg.get("user") or pg.get("username")
                pwd = pg.get("password")
                port = str(pg.get("port") or 5432)
                sslmode = pg.get("sslmode") or "require"
                if host and db and user and pwd:
                    dsn = f"postgresql://{user}:{pwd}@{host}:{port}/{db}?sslmode={sslmode}"
    except Exception:
        pass
    if not dsn:
        dsn = os.getenv("POSTGRES_URL")
        if not dsn:
            host = os.getenv("POSTGRES_HOST")
            db = os.getenv("POSTGRES_DB")
            user = os.getenv("POSTGRES_USER")
            pwd = os.getenv("POSTGRES_PASSWORD")
            port = os.getenv("POSTGRES_PORT") or "5432"
            sslmode = os.getenv("POSTGRES_SSLMODE") or "require"
            if host and db and user and pwd:
                dsn = f"postgresql://{user}:{pwd}@{host}:{port}/{db}?sslmode={sslmode}"
    if dsn:
        try:
            import psycopg2
            d = dsn
            if "channel_binding=" in d:
                d = d.replace("channel_binding=require", "").replace("&&", "&").replace("?&", "?").rstrip("&?")
            if "sslmode=" not in dsn:
                d = (d + "&sslmode=require") if ("?" in d) else (d + "?sslmode=require")
            conn = psycopg2.connect(d)
            conn.autocommit = True
            return conn, "pg"
        except Exception:
            st.warning("No se pudo conectar a Postgres; se usará SQLite local.")
    return sqlite3.connect(DB_PATH), "sqlite"

def _q(engine, sql):
    return sql.replace("?", "%s") if engine == "pg" else sql

def init_db():
    conn, engine = get_conn()
    cur = conn.cursor()

    if engine == "sqlite":
        cur.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE,
            password TEXT,
            rol TEXT,
            foto_rostro TEXT,
            foto_data BLOB
        )""")
        try:
            cols = [c[1] for c in cur.execute("PRAGMA table_info(usuarios)").fetchall()]
            if "foto_data" not in cols:
                cur.execute("ALTER TABLE usuarios ADD COLUMN foto_data BLOB")
        except sqlite3.OperationalError:
            pass
    else:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS usuarios (
            id SERIAL PRIMARY KEY,
            username TEXT UNIQUE,
            password TEXT,
            rol TEXT,
            foto_rostro TEXT,
            foto_data BYTEA
        )""")
        try:
            cur.execute("ALTER TABLE usuarios ADD COLUMN IF NOT EXISTS foto_data BYTEA")
        except Exception:
            pass

    if engine == "sqlite":
        cur.execute("""
        CREATE TABLE IF NOT EXISTS historial (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT NOT NULL,
            usuario TEXT NOT NULL,
            accion TEXT NOT NULL,
            detalles TEXT
        )""")
    else:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS historial (
            id SERIAL PRIMARY KEY,
            fecha TEXT NOT NULL,
            usuario TEXT NOT NULL,
            accion TEXT NOT NULL,
            detalles TEXT
        )""")

    if engine == "sqlite":
        cur.execute("""
        CREATE TABLE IF NOT EXISTS imagenes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ruta TEXT,
            fecha TEXT NOT NULL,
            usuario TEXT,
            data BLOB
        )""")
        try:
            cols = [c[1] for c in cur.execute("PRAGMA table_info(imagenes)").fetchall()]
            if "data" not in cols:
                cur.execute("ALTER TABLE imagenes ADD COLUMN data BLOB")
        except sqlite3.OperationalError:
            pass
    else:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS imagenes (
            id SERIAL PRIMARY KEY,
            ruta TEXT,
            fecha TEXT NOT NULL,
            usuario TEXT,
            data BYTEA
        )""")
        try:
            cur.execute("ALTER TABLE imagenes ADD COLUMN IF NOT EXISTS data BYTEA")
        except Exception:
            pass

    if engine == "sqlite":
        cur.execute("""
        CREATE TABLE IF NOT EXISTS eventos_luz (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT NOT NULL,
            accion TEXT NOT NULL,
            usuario TEXT
        )""")
    else:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS eventos_luz (
            id SERIAL PRIMARY KEY,
            fecha TEXT NOT NULL,
            accion TEXT NOT NULL,
            usuario TEXT
        )""")

    if engine == "sqlite":
        cur.execute("""
        CREATE TABLE IF NOT EXISTS eventos_alerta (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fecha TEXT NOT NULL,
            tipo TEXT NOT NULL,
            descripcion TEXT,
            usuario TEXT
        )""")
    else:
        cur.execute("""
        CREATE TABLE IF NOT EXISTS eventos_alerta (
            id SERIAL PRIMARY KEY,
            fecha TEXT NOT NULL,
            tipo TEXT NOT NULL,
            descripcion TEXT,
            usuario TEXT
        )""")

    # Asegurar columna foto_rostro en BD existente
    if engine == "sqlite":
        try:
            cols = [c[1] for c in cur.execute("PRAGMA table_info(usuarios)").fetchall()]
            if "foto_rostro" not in cols:
                cur.execute("ALTER TABLE usuarios ADD COLUMN foto_rostro TEXT")
        except sqlite3.OperationalError:
            pass

    # Usuarios de prueba
    if engine == "sqlite":
        cur.execute("INSERT OR IGNORE INTO usuarios (username, password, rol) VALUES (?, ?, ?)",
                    ("admin", "1234", "Administrador"))
        cur.execute("INSERT OR IGNORE INTO usuarios (username, password, rol) VALUES (?, ?, ?)",
                    ("cliente", "abcd", "Usuario"))
    else:
        cur.execute(_q(engine, "INSERT INTO usuarios (username, password, rol) VALUES (?, ?, ?) ON CONFLICT (username) DO NOTHING"),
                    ("admin", "1234", "Administrador"))
        cur.execute(_q(engine, "INSERT INTO usuarios (username, password, rol) VALUES (?, ?, ?) ON CONFLICT (username) DO NOTHING"),
                    ("cliente", "abcd", "Usuario"))

    if engine == "sqlite":
        conn.commit()
    conn.close()

# --------------------------
# Helpers
# --------------------------
def guardar_evento(usuario, accion, detalles=""):
    conn, engine = get_conn()
    cur = conn.cursor()
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(_q(engine, "INSERT INTO historial (fecha, usuario, accion, detalles) VALUES (?, ?, ?, ?)"),
                (fecha, usuario, accion, detalles))
    if engine == "sqlite":
        conn.commit()
    conn.close()

def guardar_imagen(img_bgr, usuario=None):
    conn, engine = get_conn()
    cur = conn.cursor()
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = None
    img_bytes = cv2.imencode('.jpg', img_bgr)[1].tobytes()
    if engine == "sqlite":
        if not os.path.exists(CAPTURAS_DIR):
            os.makedirs(CAPTURAS_DIR)
        filename = f"{CAPTURAS_DIR}/captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, img_bgr)
    try:
        cur.execute(_q(engine, "INSERT INTO imagenes (ruta, fecha, usuario, data) VALUES (?, ?, ?, ?)"),
                    (filename, fecha, usuario, sqlite3.Binary(img_bytes) if engine == "sqlite" else img_bytes))
    except Exception:
        cur.execute(_q(engine, "INSERT INTO imagenes (ruta, fecha, usuario) VALUES (?, ?, ?)"),
                    (filename, fecha, usuario))
    if engine == "sqlite":
        conn.commit()
    conn.close()
    guardar_evento(usuario or "sistema", "Captura de imagen", filename or "bytes")
    return filename or "bytes"

def notificar(msg, tipo="info"):
    toast = getattr(st, "toast", None)
    if toast:
        toast(msg)
    else:
        if tipo == "error":
            st.error(msg)
        elif tipo in ("warn", "alerta"):
            st.warning(msg)
        else:
            st.info(msg)
    if st.session_state.get("alert_show_sidebar"):
        if tipo == "error":
            st.sidebar.error(msg)
        elif tipo in ("warn", "alerta"):
            st.sidebar.warning(msg)
        else:
            st.sidebar.info(msg)
    if st.session_state.get("alert_sound"):
        sr = 16000
        t = np.linspace(0, 0.2, int(sr * 0.2), False)
        sig = np.sin(2 * np.pi * 880 * t)
        sig = np.clip(sig, -1, 1)
        data = (sig * 32767).astype(np.int16).tobytes()
        import struct, io
        nch = 1
        sw = 2
        br = sr * nch * sw
        ba = nch * sw
        buf = io.BytesIO()
        buf.write(b"RIFF")
        buf.write(struct.pack("<I", 36 + len(data)))
        buf.write(b"WAVEfmt ")
        buf.write(struct.pack("<I", 16))
        buf.write(struct.pack("<H", 1))
        buf.write(struct.pack("<H", nch))
        buf.write(struct.pack("<I", sr))
        buf.write(struct.pack("<I", br))
        buf.write(struct.pack("<H", ba))
        buf.write(struct.pack("<H", sw * 8))
        buf.write(b"data")
        buf.write(struct.pack("<I", len(data)))
        buf.write(data)
        b64 = base64.b64encode(buf.getvalue()).decode()
        st.markdown(f"<audio autoplay hidden src='data:audio/wav;base64,{b64}'></audio>", unsafe_allow_html=True)

def _bytes_to_bgr(img_bytes):
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img

def _detect_face_roi(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    x, y, w, h = faces[0]
    roi = img_bgr[y:y+h, x:x+w]
    return roi

def _compare_histogram(imgA_bgr, imgB_bgr):
    A = cv2.cvtColor(cv2.resize(imgA_bgr, (256, 256)), cv2.COLOR_BGR2HSV)
    B = cv2.cvtColor(cv2.resize(imgB_bgr, (256, 256)), cv2.COLOR_BGR2HSV)
    histA = cv2.calcHist([A], [0,1], None, [50,50], [0,180,0,256])
    histB = cv2.calcHist([B], [0,1], None, [50,50], [0,180,0,256])
    cv2.normalize(histA, histA)
    cv2.normalize(histB, histB)
    score = cv2.compareHist(histA, histB, cv2.HISTCMP_CORREL)
    return float(score)

def comparar_rostro_con_perfil(username, img_bytes, umbral=0.0):
    conn, engine = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(_q(engine, "SELECT foto_rostro FROM usuarios WHERE username=?"), (username,))
        row = cur.fetchone()
        ruta_db = row[0] if row and row[0] else None
    except sqlite3.OperationalError:
        ruta_db = None
    conn.close()

    candidatos = []
    if ruta_db and os.path.exists(ruta_db):
        candidatos.append(ruta_db)
    if os.path.exists(FOTOS_ROSTRO_DIR):
        for f in os.listdir(FOTOS_ROSTRO_DIR):
            fl = f.lower()
            if fl.endswith((".jpg", ".jpeg", ".png")) and (f.startswith(username) or "rostro" in fl):
                p = os.path.join(FOTOS_ROSTRO_DIR, f)
                if os.path.exists(p):
                    candidatos.append(p)
    vistos = set()
    candidatos = [c for c in candidatos if not (c in vistos or vistos.add(c))]

    img_input = _bytes_to_bgr(img_bytes)
    if img_input is None:
        return False, 0.0, "Imagen de entrada inválida"
    roi_input = _detect_face_roi(img_input)
    if roi_input is None:
        roi_input = img_input

    mejor = 0.0
    for ruta in candidatos:
        img_perfil = cv2.imread(ruta)
        if img_perfil is None:
            continue
        roi_perfil = _detect_face_roi(img_perfil)
        if roi_perfil is None:
            roi_perfil = img_perfil
        score = _compare_histogram(roi_perfil, roi_input)
        if score > mejor:
            mejor = score
    ok = mejor >= umbral
    return ok, mejor, "Coincidencia" if ok else "Rostro no coincide"

def guardar_imagen_bytes(img_bytes, usuario=None):
    conn, engine = get_conn()
    cur = conn.cursor()
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    filename = None
    if engine == "sqlite":
        if not os.path.exists(CAPTURAS_DIR):
            os.makedirs(CAPTURAS_DIR)
        filename = f"{CAPTURAS_DIR}/captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        with open(filename, "wb") as f:
            f.write(img_bytes)
    try:
        cur.execute(_q(engine, "INSERT INTO imagenes (ruta, fecha, usuario, data) VALUES (?, ?, ?, ?)"),
                    (filename, fecha, usuario, sqlite3.Binary(img_bytes) if engine == "sqlite" else img_bytes))
    except Exception:
        cur.execute(_q(engine, "INSERT INTO imagenes (ruta, fecha, usuario) VALUES (?, ?, ?)"),
                    (filename, fecha, usuario))
    if engine == "sqlite":
        conn.commit()
    conn.close()
    guardar_evento(usuario or "sistema", "Captura de imagen", filename or "bytes")
    return filename or "bytes"

def guardar_foto_perfil_bytes(username, img_bytes):
    if not os.path.exists(FOTOS_ROSTRO_DIR):
        os.makedirs(FOTOS_ROSTRO_DIR)
    ruta = f"{FOTOS_ROSTRO_DIR}/{username}_rostro.jpg"
    with open(ruta, "wb") as f:
        f.write(img_bytes)
    conn, engine = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(_q(engine, "UPDATE usuarios SET foto_rostro=?, foto_data=? WHERE username=?"),
                    (ruta, sqlite3.Binary(img_bytes) if engine == "sqlite" else img_bytes, username))
    except Exception:
        try:
            if engine == "sqlite":
                cur.execute("ALTER TABLE usuarios ADD COLUMN foto_rostro TEXT")
                cur.execute("ALTER TABLE usuarios ADD COLUMN foto_data BLOB")
                conn.commit()
            else:
                cur.execute("ALTER TABLE usuarios ADD COLUMN IF NOT EXISTS foto_rostro TEXT")
                cur.execute("ALTER TABLE usuarios ADD COLUMN IF NOT EXISTS foto_data BYTEA")
            cur.execute(_q(engine, "UPDATE usuarios SET foto_rostro=?, foto_data=? WHERE username=?"),
                        (ruta, sqlite3.Binary(img_bytes) if engine == "sqlite" else img_bytes, username))
        except Exception:
            pass
    if engine == "sqlite":
        conn.commit()
    conn.close()
    guardar_evento(username, "Actualizó su foto de rostro", ruta)
    return ruta

# duplicado eliminado: usar la versión superior con soporte de DB externa

def registrar_evento_luz(usuario, accion):
    conn, engine = get_conn()
    cur = conn.cursor()
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(_q(engine, "INSERT INTO eventos_luz (fecha, accion, usuario) VALUES (?, ?, ?)"),
                (fecha, accion, usuario))
    if engine == "sqlite":
        conn.commit()
    conn.close()
    guardar_evento(usuario, f"Luz - {accion}", "")

def registrar_alerta(usuario, tipo, descripcion):
    conn, engine = get_conn()
    cur = conn.cursor()
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute(_q(engine, "INSERT INTO eventos_alerta (fecha, tipo, descripcion, usuario) VALUES (?, ?, ?, ?)"),
                (fecha, tipo, descripcion, usuario))
    if engine == "sqlite":
        conn.commit()
    conn.close()
    guardar_evento(usuario, f"Alerta - {tipo}", descripcion)

# --------------------------
# Autenticación
# --------------------------
def login(username, password):
    conn, engine = get_conn()
    cur = conn.cursor()

    cur.execute(_q(engine, "SELECT rol FROM usuarios WHERE username=? AND password=?"), (username, password))
    res = cur.fetchone()

    conn.close()
    return res[0] if res else None

# --------------------------
# PERFIL (A1)
# --------------------------
def perfil_usuario(username):
    st.header("Mi Perfil")

    conn, engine = get_conn()
    cur = conn.cursor()

    try:
        if engine == "sqlite":
            cols = [c[1] for c in cur.execute("PRAGMA table_info(usuarios)").fetchall()]
            if "foto_rostro" not in cols:
                cur.execute("ALTER TABLE usuarios ADD COLUMN foto_rostro TEXT")
                conn.commit()
    except sqlite3.OperationalError:
        pass

    try:
        cur.execute(_q(engine, "SELECT foto_rostro, foto_data FROM usuarios WHERE username=?"), (username,))
        data = cur.fetchone()
        foto_actual = data[0] if data else None
        foto_bytes = data[1] if data and len(data) > 1 else None
    except sqlite3.OperationalError:
        try:
            if engine == "sqlite":
                cur.execute("ALTER TABLE usuarios ADD COLUMN foto_rostro TEXT")
                cur.execute("ALTER TABLE usuarios ADD COLUMN foto_data BLOB")
                conn.commit()
            cur.execute(_q(engine, "SELECT foto_rostro, foto_data FROM usuarios WHERE username=?"), (username,))
            data = cur.fetchone()
            foto_actual = data[0] if data else None
            foto_bytes = data[1] if data and len(data) > 1 else None
        except sqlite3.OperationalError:
            foto_actual = None
            foto_bytes = None
    conn.close()

    st.subheader("Foto de rostro guardada:")
    if foto_actual and os.path.exists(foto_actual):
        st.image(foto_actual, width=250)
    elif foto_bytes:
        st.image(foto_bytes, width=250)
    else:
        st.info("No tienes una foto registrada todavía.")

    st.subheader("Subir nueva foto de rostro")

    nueva_foto = st.file_uploader("Selecciona una imagen", type=["jpg", "png", "jpeg"])
    if "show_cam_perfil" not in st.session_state:
        st.session_state.show_cam_perfil = False
    if not st.session_state.show_cam_perfil:
        if st.button("Activar cámara"):
            st.session_state.show_cam_perfil = True
        foto_cam = None
    else:
        foto_cam = st.camera_input("Tomar foto para perfil", key="perfil_cam_input")

    if nueva_foto:
        if not os.path.exists(FOTOS_ROSTRO_DIR):
            os.makedirs(FOTOS_ROSTRO_DIR)

        ruta = f"{FOTOS_ROSTRO_DIR}/{username}_rostro.jpg"
        with open(ruta, "wb") as f:
            f.write(nueva_foto.read())

        conn, engine = get_conn()
        cur = conn.cursor()
        try:
            cur.execute(_q(engine, "UPDATE usuarios SET foto_rostro=? WHERE username=?"), (ruta, username))
        except sqlite3.OperationalError:
            try:
                if engine == "sqlite":
                    cur.execute("ALTER TABLE usuarios ADD COLUMN foto_rostro TEXT")
                    conn.commit()
                cur.execute(_q(engine, "UPDATE usuarios SET foto_rostro=? WHERE username=?"), (ruta, username))
            except sqlite3.OperationalError:
                pass
        if engine == "sqlite":
            conn.commit()
        conn.close()

        guardar_evento(username, "Actualizó su foto de rostro", ruta)

        st.success("Foto guardada exitosamente.")
        st.image(ruta, width=250)

    if foto_cam is not None:
        ruta = guardar_foto_perfil_bytes(username, foto_cam.getvalue())
        st.success("Foto guardada exitosamente.")
        st.image(ruta, width=250)

# --------------------------
# Interfaces UI
# --------------------------
def mostrar_historial():
    st.header("Historial de actividades")
    import pandas as pd
    conn, _ = get_conn()
    df = pd.read_sql_query("SELECT fecha, usuario, accion, detalles FROM historial ORDER BY fecha DESC", conn)
    conn.close()

    if df.empty:
        st.warning("No hay actividades registradas aún.")
        return

    st.dataframe(df, height=400)

def controlar_luces(usuario):
    st.header("Control de luces")
    if "luz_on" not in st.session_state:
        st.session_state.luz_on = False
    colz1, colz2 = st.columns(2)
    zona = colz1.selectbox("Zona", ["Jardín", "Terraza"], index=0)
    modo = colz2.selectbox("Modo", ["Manual", "Automático"], index=0)
    intensidad = st.slider("Intensidad (%)", 0, 100, 50)
    colp1, colp2 = st.columns(2)
    if colp1.button("⏻ Encender todas las luces", key="btn_power_on"):
        st.session_state.luz_on = True
        registrar_evento_luz(usuario, f"Encendido {zona} {intensidad}% {modo}")
        st.success("Luz encendida")
    if colp2.button("Apagar todas las luces", key="btn_power_off"):
        st.session_state.luz_on = False
        registrar_evento_luz(usuario, f"Apagado {zona} {modo}")
        st.info("Luz apagada")
    col1, col2 = st.columns(2)
    if col1.button("Encender"):
        registrar_evento_luz(usuario, f"Encendido {zona} {intensidad}% {modo}")
        st.success(f"Encendido {zona} a {intensidad}%")
    if col2.button("Apagar"):
        registrar_evento_luz(usuario, f"Apagado {zona} {modo}")
        st.info(f"Apagado {zona}")

    st.divider()
    st.subheader("Eventos recientes")
    import pandas as pd
    conn, _ = get_conn()
    df = pd.read_sql_query("SELECT fecha, accion, usuario FROM eventos_luz ORDER BY fecha DESC LIMIT 10", conn)
    conn.close()
    st.dataframe(df, width='stretch')

def simular_alerta(usuario):
    st.header("Alertas")
    with st.form("alert_form"):
        tipo = st.selectbox("Tipo de alerta", ["Movimiento detectado", "Intruso", "Sensor fallo"])
        descripcion = st.text_area("Descripción (opcional)")
        submitted = st.form_submit_button("Registrar alerta")
    if submitted:
        registrar_alerta(usuario, tipo, descripcion)
        nivel = "alerta" if tipo == "Intruso" else ("error" if tipo == "Sensor fallo" else "warn")
        notificar(f"{tipo}", nivel)
        token = None
        chat_id = None
        try:
            token = st.secrets.get("telegram_token")
            chat_id = st.secrets.get("telegram_chat_id")
        except Exception:
            pass
        if not token or not chat_id:
            token = os.getenv("TELEGRAM_TOKEN")
            chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if token and chat_id:
            try:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                requests.post(url, data={"chat_id": chat_id, "text": f"Alerta: {tipo} - {descripcion if descripcion else ''}"}, timeout=10)
            except Exception:
                pass

    st.divider()
    st.subheader("Simulación automática")
    if "sim_alert_on" not in st.session_state:
        st.session_state.sim_alert_on = False
    if "alert_interval" not in st.session_state:
        st.session_state.alert_interval = 20
    if "next_alert_ts" not in st.session_state:
        st.session_state.next_alert_ts = time.time() + st.session_state.alert_interval
    opt1, opt2 = st.columns(2)
    if "alert_show_sidebar" not in st.session_state:
        st.session_state.alert_show_sidebar = False
    if "alert_sound" not in st.session_state:
        st.session_state.alert_sound = False
    opt1.toggle("Mostrar notificación en sidebar", value=st.session_state.alert_show_sidebar, key="alert_show_sidebar")
    opt2.toggle("Sonido", value=st.session_state.alert_sound, key="alert_sound")
    if st.button("Probar Telegram"):
        token = st.session_state.get("telegram_token")
        chat_id = st.session_state.get("telegram_chat_id")
        if not token or not chat_id:
            try:
                token = token or st.secrets.get("telegram_token")
                chat_id = chat_id or st.secrets.get("telegram_chat_id")
            except Exception:
                pass
        if not token or not chat_id:
            token = os.getenv("TELEGRAM_TOKEN")
            chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            st.warning("Configura telegram_token y telegram_chat_id en Secrets o variables de entorno.")
        else:
            try:
                url = f"https://api.telegram.org/bot{token}/sendMessage"
                r = requests.post(url, data={"chat_id": chat_id, "text": "Prueba de Telegram desde la app"}, timeout=10)
                if r.status_code == 200:
                    st.success("Prueba enviada a Telegram")
                else:
                    st.error(f"No se pudo enviar (status {r.status_code})")
            except Exception:
                st.error("Error al enviar a Telegram")
    col1, col2 = st.columns([2,1])
    intervalo = col1.number_input("Intervalo (segundos)", min_value=5, max_value=3600, value=int(st.session_state.alert_interval), step=5)
    if int(intervalo) != st.session_state.alert_interval:
        st.session_state.alert_interval = int(intervalo)
    if not st.session_state.sim_alert_on:
        if col2.button("Iniciar simulación"):
            st.session_state.sim_alert_on = True
            st.session_state.next_alert_ts = time.time() + st.session_state.alert_interval
    else:
        if col2.button("Detener simulación"):
            st.session_state.sim_alert_on = False
    if st.session_state.sim_alert_on:
        st.caption("Genera una alerta aleatoria automáticamente.")
        restante = max(0, int(st.session_state.next_alert_ts - time.time()))
        st.write(f"Siguiente en ~ {restante} s")
        if time.time() >= st.session_state.next_alert_ts:
            tipo_auto = random.choice(["Movimiento detectado", "Sensor fallo", "Intruso"])
            zona = random.choice(["Jardín", "Terraza"])
            registrar_alerta(usuario, tipo_auto, f"{zona} - simulado")
            nivel = "alerta" if tipo_auto == "Intruso" else ("error" if tipo_auto == "Sensor fallo" else "warn")
            notificar(f"{tipo_auto} en {zona}", nivel)
            token = st.session_state.get("telegram_token")
            chat_id = st.session_state.get("telegram_chat_id")
            if not token or not chat_id:
                try:
                    token = token or st.secrets.get("telegram_token")
                    chat_id = chat_id or st.secrets.get("telegram_chat_id")
                except Exception:
                    pass
            if not token or not chat_id:
                token = os.getenv("TELEGRAM_TOKEN")
                chat_id = os.getenv("TELEGRAM_CHAT_ID")
            if token and chat_id:
                try:
                    url = f"https://api.telegram.org/bot{token}/sendMessage"
                    requests.post(url, data={"chat_id": chat_id, "text": f"Alerta automática: {tipo_auto} en {zona}"}, timeout=10)
                except Exception:
                    pass
            st.session_state.next_alert_ts = time.time() + st.session_state.alert_interval
        time.sleep(1)
        st.rerun()

    st.divider()
    import pandas as pd
    conn, _ = get_conn()
    df = pd.read_sql_query("SELECT fecha, tipo, descripcion FROM eventos_alerta ORDER BY fecha DESC LIMIT 10", conn)
    conn.close()
    st.dataframe(df, width='stretch')

def camara_interface(usuario):
    st.header("Cámara")
    image_file = st.camera_input("Tomar foto")
    if image_file is not None:
        ruta = guardar_imagen_bytes(image_file.getvalue(), usuario)
        st.success(f"Foto guardada: {ruta}")
        st.image(image_file)
    else:
        archivo = st.file_uploader("Subir imagen", type=["jpg", "jpeg", "png"])
        if archivo is not None:
            ruta = guardar_imagen_bytes(archivo.getvalue(), usuario)
            st.success(f"Foto guardada: {ruta}")
            st.image(archivo)
    with st.expander("Captura avanzada (OpenCV)"):
        idx = st.number_input("Índice de cámara", min_value=0, max_value=5, step=1, value=0)
        if st.button("Probar cámara"):
            cap = cv2.VideoCapture(int(idx))
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    ruta = guardar_imagen(frame, usuario)
                    st.success(f"Foto guardada: {ruta}")
                    st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    st.error("No se pudo capturar imagen.")
                cap.release()
            else:
                st.error("No se pudo abrir la cámara. Revisa permisos y dispositivo.")

# --------------------------
# App principal
# --------------------------
def main():
    st.set_page_config(page_title="Luminest", layout="wide")
    init_db()

    if "telegram_token" not in st.session_state:
        tk = None
        cid = None
        try:
            tk = st.secrets.get("telegram_token")
            cid = st.secrets.get("telegram_chat_id")
        except Exception:
            pass
        if not tk:
            tk = os.getenv("TELEGRAM_TOKEN")
        if not cid:
            cid = os.getenv("TELEGRAM_CHAT_ID")
        if tk:
            st.session_state.telegram_token = tk
        if cid:
            st.session_state.telegram_chat_id = cid

    st.markdown("""
        <style>
        .stApp h1 { margin-bottom: 0.5rem; }
        </style>
    """, unsafe_allow_html=True)
    st.title("Luminest")
    try:
        c, eng = get_conn()
        c.close()
        st.caption(f"Almacenamiento: {'Postgres' if eng == 'pg' else 'SQLite'}")
    except Exception:
        st.caption("Almacenamiento: SQLite")

    if "usuario" not in st.session_state:
        st.session_state.usuario = None
        st.session_state.rol = None

    if st.session_state.usuario is None:
        st.subheader("Iniciar sesión")
        tab1, tab2 = st.tabs(["Usuario/Contraseña", "Rostro"])
        with tab1:
            username = st.text_input("Usuario")
            password = st.text_input("Contraseña", type="password")
            if st.button("Entrar"):
                rol = login(username, password)
                if rol:
                    st.session_state.usuario = username
                    st.session_state.rol = rol
                    guardar_evento(username, "Inicio de sesión", f"Rol: {rol}")
                    st.rerun()
                else:
                    st.error("Usuario o contraseña incorrectos")
        with tab2:
            st.caption("Requiere tener una foto de perfil guardada en 'Mi Perfil'.")
            face_cam = st.camera_input("Tomar foto")
            face_file = st.file_uploader("O subir imagen", type=["jpg", "jpeg", "png"])
            username_face = st.text_input("Usuario para validar rostro")
            umbral_val = 0.0
            st.caption("Si el navegador no habilita la cámara, prueba con OpenCV abajo.")
            idx = st.number_input("Índice de cámara (OpenCV)", min_value=0, max_value=5, step=1, value=0)
            if st.button("Capturar con OpenCV y entrar"):
                cap = cv2.VideoCapture(int(idx))
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        ok, score, msg = comparar_rostro_con_perfil(username_face, cv2.imencode('.jpg', frame)[1].tobytes(), umbral=umbral_val)
                        if ok:
                            rol = login(username_face, ("")) or "Usuario"
                            st.session_state.usuario = username_face
                            st.session_state.rol = rol
                            guardar_evento(username_face, "Inicio de sesión por rostro (OpenCV)", f"score={score:.2f}")
                            st.success("Autenticación por rostro exitosa")
                            st.rerun()
                        else:
                            st.error(f"{msg} (score={score:.2f})")
                    else:
                        st.error("No se pudo capturar imagen.")
                else:
                    st.error("No se pudo abrir la cámara con OpenCV. Cambia el índice o revisa permisos.")
            if st.button("Entrar con rostro"):
                img_bytes = None
                if face_cam is not None:
                    img_bytes = face_cam.getvalue()
                elif face_file is not None:
                    img_bytes = face_file.getvalue()
                if not username_face:
                    st.error("Ingresa el usuario a validar.")
                elif not img_bytes:
                    st.error("Captura o sube una imagen del rostro.")
                else:
                    ok, score, msg = comparar_rostro_con_perfil(username_face, img_bytes, umbral=umbral_val)
                    if ok:
                        rol = login(username_face, ("")) or "Usuario"
                        st.session_state.usuario = username_face
                        st.session_state.rol = rol
                        guardar_evento(username_face, "Inicio de sesión por rostro", f"score={score:.2f}")
                        st.success("Autenticación por rostro exitosa")
                        st.rerun()
                    else:
                        st.error(f"{msg} (score={score:.2f})")
        return

    # Menú lateral
    st.sidebar.write(f"**Usuario:** {st.session_state.usuario}")
    st.sidebar.write(f"**Rol:** {st.session_state.rol}")
    conn, engine = get_conn()
    cur = conn.cursor()
    try:
        cur.execute(_q(engine, "SELECT foto_rostro FROM usuarios WHERE username=?"), (st.session_state.usuario,))
        row = cur.fetchone()
        foto_sidebar = row[0] if row and row[0] else None
    except sqlite3.OperationalError:
        foto_sidebar = None
    conn.close()
    if foto_sidebar and os.path.exists(foto_sidebar):
        st.sidebar.image(foto_sidebar, width=80)

    opcion = st.sidebar.selectbox("Menú", [
        "Mi Perfil",
        "Control de luces",
        "Alertas",
        "Cámara",
        "Historial",
        "Cerrar sesión"
    ])

    if opcion == "Mi Perfil":
        perfil_usuario(st.session_state.usuario)

    elif opcion == "Control de luces":
        controlar_luces(st.session_state.usuario)

    elif opcion == "Alertas":
        simular_alerta(st.session_state.usuario)

    elif opcion == "Cámara":
        camara_interface(st.session_state.usuario)

    elif opcion == "Historial":
        if st.session_state.rol == "Administrador":
            mostrar_historial()
        else:
            st.warning("Acceso denegado: Solo administradores.")

    elif opcion == "Cerrar sesión":
        guardar_evento(st.session_state.usuario, "Cierre de sesión", "")
        st.session_state.usuario = None
        st.session_state.rol = None
        st.rerun()

if __name__ == "__main__":
    main()
