"""
app.py
Flask-based Face Recognition Attendance System
Admin login -> Register employees (capture face via webcam) -> Live attendance via face recognition
"""
import os
import time
import platform
import cv2
import numpy as np
from functools import wraps
from flask import (
    Flask, render_template, request, redirect, url_for,
    session, Response, jsonify, flash
)

import database as db
import face_utils as fu

app = Flask(__name__)
app.secret_key = "change-this-secret-key-in-production"

ADMIN_USERNAME = "muzammil"
ADMIN_PASSWORD = "muzammil182921"

db.init_db()

# ---------- Shared in-memory state ----------
capture_progress = {}     # employee_id -> {"count": int, "done": bool}
last_marked = {}          # employee_id -> timestamp (to avoid spamming DB checks every frame)
camera = None             # single shared VideoCapture instance
CAMERA_INDEX = 0          # change to 1 if your webcam is not the default device


def _open_camera(index):
    """Try the best backend for the current OS, with a plain fallback."""
    if platform.system() == "Windows":
        cam = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cam.isOpened():
            cam.release()
            cam = cv2.VideoCapture(index, cv2.CAP_MSMF)
    else:
        cam = cv2.VideoCapture(index)
    return cam


def get_camera():
    global camera
    if camera is None or not camera.isOpened():
        camera = _open_camera(CAMERA_INDEX)
        if camera.isOpened():
            # warm-up: first few frames from a freshly opened camera are often empty/black
            for _ in range(5):
                camera.read()
                time.sleep(0.02)
    return camera


def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None


def error_frame_bytes(message):
    """Build a single JPEG frame carrying an error message, so the <img> tag
    always shows something instead of a broken image icon."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "CAMERA ERROR", (150, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    for i, line in enumerate(message.split("\n")):
        cv2.putText(img, line, (60, 260 + i * 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 255), 1)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def mjpeg_wrap(frame_bytes):
    return (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


# ---------------- Auth ----------------

def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return wrapper


@app.route("/", methods=["GET"])
def index():
    return redirect(url_for("dashboard") if session.get("logged_in") else url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("dashboard"))
        error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/logout")
def logout():
    session.clear()
    release_camera()
    return redirect(url_for("login"))


# ---------------- Dashboard ----------------

@app.route("/dashboard")
@login_required
def dashboard():
    stats = db.get_stats()
    today_logs = db.get_today_attendance()
    pending = db.get_pending_approvals()
    return render_template("dashboard.html", stats=stats, today_logs=today_logs, pending=pending)


@app.route("/employees/approve/<int:employee_id>", methods=["POST"])
@login_required
def approve_employee(employee_id):
    db.approve_employee(employee_id)
    flash("Employee approved — their face will now be recognized in Live Attendance.", "success")
    return redirect(request.referrer or url_for("dashboard"))


@app.route("/employees/revoke/<int:employee_id>", methods=["POST"])
@login_required
def revoke_employee(employee_id):
    db.revoke_approval(employee_id)
    flash("Approval revoked — this employee will no longer be recognized until re-approved.", "success")
    return redirect(request.referrer or url_for("employees"))


@app.route("/api/stats")
@login_required
def api_stats():
    return jsonify(db.get_stats())


# ---------------- Employees ----------------

@app.route("/employees")
@login_required
def employees():
    all_employees = db.get_all_employees()
    return render_template("employees.html", employees=all_employees)


@app.route("/employees/delete/<int:employee_id>", methods=["POST"])
@login_required
def delete_employee(employee_id):
    db.delete_employee(employee_id)
    flash("Employee removed.", "success")
    return redirect(url_for("employees"))


# ---------------- Registration (new person) ----------------

@app.route("/register", methods=["GET", "POST"])
@login_required
def register():
    departments = db.get_all_departments()
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        department_id = request.form.get("department_id")
        if not name or not department_id:
            flash("Name and department are required.", "error")
            return render_template("register.html", departments=departments)

        new_id = db.add_employee(name, int(department_id))
        capture_progress[new_id] = {"count": 0, "done": False}
        return redirect(url_for("capture_page", employee_id=new_id))

    return render_template("register.html", departments=departments)


@app.route("/departments/add", methods=["POST"])
@login_required
def add_department():
    name = request.form.get("dept_name", "").strip()
    if name:
        ok = db.add_department(name)
        flash("Department added." if ok else "Department already exists.",
              "success" if ok else "error")
    return redirect(url_for("register"))


@app.route("/capture/<int:employee_id>")
@login_required
def capture_page(employee_id):
    employee = db.get_employee(employee_id)
    if employee is None:
        return redirect(url_for("register"))
    return render_template("capture.html", employee=employee, needed=fu.SAMPLES_NEEDED)


def gen_capture_frames(employee_id):
    cam = get_camera()
    capture_progress[employee_id] = {"count": 0, "done": False}

    if not cam.isOpened():
        msg = ("No webcam detected.\n"
               "Close Zoom/Teams/other apps using the camera,\n"
               "check CAMERA_INDEX in app.py, then refresh this page.")
        for _ in range(1000):
            yield mjpeg_wrap(error_frame_bytes(msg))
            time.sleep(0.5)
        return

    sample_index = 0
    fail_count = 0

    while sample_index < fu.SAMPLES_NEEDED:
        success, frame = cam.read()
        if not success:
            fail_count += 1
            if fail_count > 30:
                yield mjpeg_wrap(error_frame_bytes("Lost connection to webcam.\nRefresh the page to retry."))
                break
            time.sleep(0.05)
            continue
        fail_count = 0

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = fu.detect_faces(gray)

        for (x, y, w, h) in faces:
            sample_index += 1
            fu.save_face_sample(employee_id, gray[y:y + h, x:x + w], sample_index)
            capture_progress[employee_id]["count"] = sample_index

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 155), 2)
            cv2.putText(frame, f"Capturing {sample_index}/{fu.SAMPLES_NEEDED}",
                        (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 155), 2)
            break  # one face per frame is enough

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")

        if sample_index >= fu.SAMPLES_NEEDED:
            break

    capture_progress[employee_id]["done"] = True
    db.update_samples_captured(employee_id, sample_index)
    release_camera()


@app.route("/capture_feed/<int:employee_id>")
@login_required
def capture_feed(employee_id):
    return Response(gen_capture_frames(employee_id),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/capture_status/<int:employee_id>")
@login_required
def capture_status(employee_id):
    progress = capture_progress.get(employee_id, {"count": 0, "done": False})
    return jsonify(progress)


@app.route("/train/<int:employee_id>", methods=["POST"])
@login_required
def train(employee_id):
    ok, message = fu.train_model()
    if ok:
        db.mark_trained(employee_id)
        flash(f"Face data trained! {message} Now waiting for admin approval on the Dashboard before this person can be recognized.", "success")
    else:
        flash(f"Training failed: {message}", "error")
    return redirect(url_for("employees"))


# ---------------- Live Attendance ----------------

@app.route("/attendance")
@login_required
def attendance_page():
    return render_template("attendance_live.html")


def gen_attendance_frames():
    cam = get_camera()
    recognizer = fu.load_recognizer()
    trained_map = {e["id"]: e for e in db.get_trained_employees()}

    if not cam.isOpened():
        msg = ("No webcam detected.\n"
               "Close Zoom/Teams/other apps using the camera,\n"
               "check CAMERA_INDEX in app.py, then refresh this page.")
        for _ in range(1000):
            yield mjpeg_wrap(error_frame_bytes(msg))
            time.sleep(0.5)
        return

    if recognizer is None:
        msg = "No trained faces yet.\nGo to 'Register Person' and add\nat least one employee first."
        for _ in range(1000):
            yield mjpeg_wrap(error_frame_bytes(msg))
            time.sleep(0.5)
        return

    fail_count = 0
    while True:
        success, frame = cam.read()
        if not success:
            fail_count += 1
            if fail_count > 30:
                yield mjpeg_wrap(error_frame_bytes("Lost connection to webcam.\nRefresh the page to retry."))
                break
            time.sleep(0.05)
            continue
        fail_count = 0

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = fu.detect_faces(gray)

        for (x, y, w, h) in faces:
            employee_id, confidence = fu.recognize_face(recognizer, gray[y:y + h, x:x + w])

            if employee_id is not None and employee_id in trained_map:
                emp = trained_map[employee_id]
                color = (0, 255, 155)
                label = f"{emp['name']} | {emp['department_name']} ({confidence:.0f})"

                now = time.time()
                if now - last_marked.get(employee_id, 0) > 5:
                    last_marked[employee_id] = now
                    newly_marked = db.mark_attendance(employee_id)
                    if newly_marked:
                        label += "  [MARKED]"
            else:
                color = (0, 60, 255)
                conf_txt = f" ({confidence:.0f})" if confidence is not None else ""
                label = f"Unknown{conf_txt}"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(frame, (x, y - 28), (x + max(w, 220), y), color, -1)
            cv2.putText(frame, label, (x + 4, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            cv2.putText(frame, time.strftime("%H:%M:%S"), (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        ret, buffer = cv2.imencode(".jpg", frame)
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(gen_attendance_frames(),
                     mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/camera_check")
@login_required
def camera_check():
    """Quick diagnostic: open the camera, try to read one frame, report result."""
    cam = get_camera()
    result = {"camera_index": CAMERA_INDEX, "opened": cam.isOpened()}
    if cam.isOpened():
        success, frame = cam.read()
        result["frame_read"] = bool(success)
        result["frame_shape"] = list(frame.shape) if success else None
    return jsonify(result)


@app.route("/stop_camera", methods=["POST"])
@login_required
def stop_camera():
    release_camera()
    return jsonify({"status": "stopped"})


@app.route("/api/live_logs")
@login_required
def api_live_logs():
    logs = db.get_today_attendance()
    return jsonify([dict(r) for r in logs])


# ---------------- Reports ----------------

@app.route("/reports")
@login_required
def reports():
    date_filter = request.args.get("date") or None
    dept_filter = request.args.get("department_id") or None
    dept_filter = int(dept_filter) if dept_filter else None

    logs = db.get_attendance_filtered(date=date_filter, department_id=dept_filter)
    departments = db.get_all_departments()
    return render_template("reports.html", logs=logs, departments=departments,
                            selected_date=date_filter, selected_dept=dept_filter)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000, threaded=True)
