import streamlit as st
import sqlite3
import os
import cv2
import numpy as np
import time
import random
import base64
from datetime import datetime

DB_PATH = "sistema.db"
CAPTURAS_DIR = "capturas"
FOTOS_ROSTRO_DIR = "fotos_rostro"

# --------------------------
# Inicialización de BD
# --------------------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS usuarios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE,
        password TEXT,
        rol TEXT,
        foto_rostro TEXT
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS historial (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fecha TEXT NOT NULL,
        usuario TEXT NOT NULL,
        accion TEXT NOT NULL,
        detalles TEXT
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS imagenes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ruta TEXT NOT NULL,
        fecha TEXT NOT NULL,
        usuario TEXT
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS eventos_luz (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fecha TEXT NOT NULL,
        accion TEXT NOT NULL,
        usuario TEXT
    )""")

    cur.execute("""
    CREATE TABLE IF NOT EXISTS eventos_alerta (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        fecha TEXT NOT NULL,
        tipo TEXT NOT NULL,
        descripcion TEXT,
        usuario TEXT
    )""")

    # Asegurar columna foto_rostro en BD existente
    try:
        cols = [c[1] for c in cur.execute("PRAGMA table_info(usuarios)").fetchall()]
        if "foto_rostro" not in cols:
            cur.execute("ALTER TABLE usuarios ADD COLUMN foto_rostro TEXT")
    except sqlite3.OperationalError:
        pass

    # Usuarios de prueba
    cur.execute("INSERT OR IGNORE INTO usuarios (username, password, rol) VALUES (?, ?, ?)",
                ("admin", "1234", "Administrador"))
    cur.execute("INSERT OR IGNORE INTO usuarios (username, password, rol) VALUES (?, ?, ?)",
                ("cliente", "abcd", "Usuario"))

    conn.commit()
    conn.close()

# --------------------------
# Helpers
# --------------------------
def guardar_evento(usuario, accion, detalles=""):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("INSERT INTO historial (fecha, usuario, accion, detalles) VALUES (?, ?, ?, ?)",
                (fecha, usuario, accion, detalles))
    conn.commit()
    conn.close()

def guardar_imagen(img_bgr, usuario=None):
    if not os.path.exists(CAPTURAS_DIR):
        os.makedirs(CAPTURAS_DIR)

    filename = f"{CAPTURAS_DIR}/captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    cv2.imwrite(filename, img_bgr)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cur.execute("INSERT INTO imagenes (ruta, fecha, usuario) VALUES (?, ?, ?)",
                (filename, fecha, usuario))

    conn.commit()
    conn.close()
    guardar_evento(usuario or "sistema", "Captura de imagen", filename)
    return filename

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
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("SELECT foto_rostro FROM usuarios WHERE username=?", (username,))
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
    if not os.path.exists(CAPTURAS_DIR):
        os.makedirs(CAPTURAS_DIR)
    filename = f"{CAPTURAS_DIR}/captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    with open(filename, "wb") as f:
        f.write(img_bytes)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("INSERT INTO imagenes (ruta, fecha, usuario) VALUES (?, ?, ?)",
                (filename, fecha, usuario))
    conn.commit()
    conn.close()
    guardar_evento(usuario or "sistema", "Captura de imagen", filename)
    return filename

def guardar_foto_perfil_bytes(username, img_bytes):
    if not os.path.exists(FOTOS_ROSTRO_DIR):
        os.makedirs(FOTOS_ROSTRO_DIR)
    ruta = f"{FOTOS_ROSTRO_DIR}/{username}_rostro.jpg"
    with open(ruta, "wb") as f:
        f.write(img_bytes)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("UPDATE usuarios SET foto_rostro=? WHERE username=?", (ruta, username))
    except sqlite3.OperationalError:
        try:
            cur.execute("ALTER TABLE usuarios ADD COLUMN foto_rostro TEXT")
            conn.commit()
            cur.execute("UPDATE usuarios SET foto_rostro=? WHERE username=?", (ruta, username))
        except sqlite3.OperationalError:
            pass
    conn.commit()
    conn.close()
    guardar_evento(username, "Actualizó su foto de rostro", ruta)
    return ruta

def guardar_imagen_bytes(img_bytes, usuario=None):
    if not os.path.exists(CAPTURAS_DIR):
        os.makedirs(CAPTURAS_DIR)
    filename = f"{CAPTURAS_DIR}/captura_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    with open(filename, "wb") as f:
        f.write(img_bytes)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("INSERT INTO imagenes (ruta, fecha, usuario) VALUES (?, ?, ?)",
                (filename, fecha, usuario))
    conn.commit()
    conn.close()
    guardar_evento(usuario or "sistema", "Captura de imagen", filename)
    return filename

def registrar_evento_luz(usuario, accion):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("INSERT INTO eventos_luz (fecha, accion, usuario) VALUES (?, ?, ?)",
                (fecha, accion, usuario))
    conn.commit()
    conn.close()
    guardar_evento(usuario, f"Luz - {accion}", "")

def registrar_alerta(usuario, tipo, descripcion):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute("INSERT INTO eventos_alerta (fecha, tipo, descripcion, usuario) VALUES (?, ?, ?, ?)",
                (fecha, tipo, descripcion, usuario))
    conn.commit()
    conn.close()
    guardar_evento(usuario, f"Alerta - {tipo}", descripcion)

# --------------------------
# Autenticación
# --------------------------
def login(username, password):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute("SELECT rol FROM usuarios WHERE username=? AND password=?", (username, password))
    res = cur.fetchone()

    conn.close()
    return res[0] if res else None

# --------------------------
# PERFIL (A1)
# --------------------------
def perfil_usuario(username):
    st.header("Mi Perfil")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    try:
        cols = [c[1] for c in cur.execute("PRAGMA table_info(usuarios)").fetchall()]
        if "foto_rostro" not in cols:
            cur.execute("ALTER TABLE usuarios ADD COLUMN foto_rostro TEXT")
            conn.commit()
    except sqlite3.OperationalError:
        pass

    try:
        cur.execute("SELECT foto_rostro FROM usuarios WHERE username=?", (username,))
        data = cur.fetchone()
        foto_actual = data[0] if data else None
    except sqlite3.OperationalError:
        try:
            cur.execute("ALTER TABLE usuarios ADD COLUMN foto_rostro TEXT")
            conn.commit()
            cur.execute("SELECT foto_rostro FROM usuarios WHERE username=?", (username,))
            data = cur.fetchone()
            foto_actual = data[0] if data else None
        except sqlite3.OperationalError:
            foto_actual = None
    conn.close()

    st.subheader("Foto de rostro guardada:")
    if foto_actual and os.path.exists(foto_actual):
        st.image(foto_actual, width=250)
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

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        try:
            cur.execute("UPDATE usuarios SET foto_rostro=? WHERE username=?", (ruta, username))
        except sqlite3.OperationalError:
            try:
                cur.execute("ALTER TABLE usuarios ADD COLUMN foto_rostro TEXT")
                conn.commit()
                cur.execute("UPDATE usuarios SET foto_rostro=? WHERE username=?", (ruta, username))
            except sqlite3.OperationalError:
                pass
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
    conn = sqlite3.connect(DB_PATH)
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
    zona = colz1.selectbox("Zona", ["Sala", "Pasillo", "Jardín"], index=0)
    modo = colz2.selectbox("Modo", ["Manual", "Automático"], index=0)
    intensidad = st.slider("Intensidad (%)", 0, 100, 50)

    power_label = "⏻ Encender todas las luces" if not st.session_state.luz_on else "⏻ Apagar todas las luces"
    if st.button(power_label, key="btn_power_luz"):
        st.session_state.luz_on = not st.session_state.luz_on
        if st.session_state.luz_on:
            registrar_evento_luz(usuario, f"Encendido {zona} {intensidad}% {modo}")
            st.success("Luz encendida")
        else:
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
    conn = sqlite3.connect(DB_PATH)
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
            zona = random.choice(["Sala", "Pasillo", "Jardín"])
            registrar_alerta(usuario, tipo_auto, f"{zona} - simulado")
            nivel = "alerta" if tipo_auto == "Intruso" else ("error" if tipo_auto == "Sensor fallo" else "warn")
            notificar(f"{tipo_auto} en {zona}", nivel)
            st.session_state.next_alert_ts = time.time() + st.session_state.alert_interval
        time.sleep(1)
        st.rerun()

    st.divider()
    import pandas as pd
    conn = sqlite3.connect(DB_PATH)
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
            st.image(ruta)
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
    st.set_page_config(page_title="Sistema Inteligente - Jardín", layout="wide")
    init_db()

    st.title("Sistema Inteligente de Iluminación y Seguridad")

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
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute("SELECT foto_rostro FROM usuarios WHERE username=?", (st.session_state.usuario,))
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
