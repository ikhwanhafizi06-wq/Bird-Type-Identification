import mysql.connector
from flask import Flask, render_template, request, redirect, session, send_from_directory, send_file
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from train import train_model
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as PDFImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch


app = Flask(__name__)
app.secret_key = "bird123"

# DATABASE CONNECTION
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="bird_detection"
)

cursor = db.cursor()

# UPLOAD IMG kat Folder
UPLOAD_FOLDER = "static/uploads"

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# LOAD AI MODEL
model = tf.keras.models.load_model(
    "model/bird_model.keras"
)

class_names = [
    "Crow",
    "Duck",
    "Eagle",
    "Owl",
    "Parrot"
]

# PREDICT FUNCTION
def predict_image(image_path):

    img = Image.open(image_path).convert("RGB")
    img = img.resize((224,224))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    index = np.argmax(prediction)

    confidence = float(
        np.max(prediction)
    ) * 100

    bird = class_names[index]

    return bird, confidence

#Home
@app.route("/")
def home():
    return render_template("home.html")

# USER DASHBOARD
@app.route("/dashboard")
def dashboard():

    if "username" not in session:
        return redirect("/login")

    if session["role"] != "user":
        return redirect("/admin")

    return render_template(
        "dashboard.html",
        username=session["username"]
    )
#LOGIN
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        sql = """
        SELECT * FROM users
        WHERE username=%s AND password=%s
        """

        cursor.execute(
            sql,
            (username, password)
        )

        user = cursor.fetchone()

        if user:

            session["username"] = user[1]
            session["role"] = user[3]

            if user[3] == "admin":
                return redirect("/admin")

            else:
                return redirect("/dashboard")

        else:

            return render_template(
                "login.html",
                error="Invalid username or password"
            )

    return render_template("login.html")
    
# REGISTER
@app.route("/register", methods=["GET","POST"])
def register():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        sql = """
        INSERT INTO users(username,password)
        VALUES(%s,%s)
        """
        cursor.execute(
            sql,
            (username,password)
        )

        db.commit()

        return redirect("/login")

    return render_template(
        "register.html"
    )

# Tentukan dashbord admin atau user
@app.route("/admin")
def admin():

    if "username" not in session:
        return redirect("/login")

    if session["role"] != "admin":
        return redirect("/dashboard")

    return render_template(
        "admin.html",
        username=session["username"]
    )

# DETECTION PAGE

@app.route("/detect", methods=["GET", "POST"])
def detect():

    if "username" not in session:
        return redirect("/login")

    if request.method == "POST":

        file = request.files["image"]

        filepath = os.path.join(
            UPLOAD_FOLDER,
            file.filename
        )

        file.save(filepath)

        # AI PREDICTION
        bird, confidence = predict_image(filepath)

        # GET BIRD INFORMATION
        cursor.execute("""
            SELECT description, habitat
            FROM birds
            WHERE bird_type=%s
        """, (bird,))

        bird_info = cursor.fetchone()

        description = ""
        habitat = ""

        if bird_info:
            description = bird_info[0]
            habitat = bird_info[1]

        # SAVE DETECTION
        sql = """
            INSERT INTO detection_history
            (username, image, bird_type, confidence)
            VALUES (%s, %s, %s, %s)
        """

        value = (
            session["username"],
            filepath,
            bird,
            confidence
        )

        cursor.execute(sql, value)
        db.commit()

        # SAVE LAST DETECTION FOR PDF
        session["last_detection"] = {
            "image": filepath,
            "bird": bird,
            "confidence": round(confidence, 2),
            "description": description,
            "habitat": habitat
        }

        return render_template(
            "detect.html",
            image=filepath,
            bird=bird,
            confidence=round(confidence, 2),
            description=description,
            habitat=habitat
        )

    return render_template("detect.html")

# DOWNLOAD DETECTION RESULT AS PDF

def add_pdf_metadata(canvas, doc):
    canvas.setTitle("Bird Type Identification - Bird Detection Report")
    canvas.setAuthor("Bird Type Identification")
    canvas.setSubject("Bird Detection Result")


@app.route("/download-result")
def download_result():

    if "username" not in session:
        return redirect("/login")

    if "last_detection" not in session:
        return redirect("/detect")

    result = session["last_detection"]

    buffer = BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=20,
        textColor="#0879c9",
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "HeadingStyle",
        parent=styles["Heading2"],
        fontSize=13,
        textColor="#0879c9",
        spaceAfter=5
    )

    normal_style = ParagraphStyle(
        "NormalStyle",
        parent=styles["Normal"],
        fontSize=11,
        leading=16
    )

    elements = []

    elements.append(
        Paragraph(
            "BIRD TYPE IDENTIFICATION",
            title_style
        )
    )

    elements.append(
        Paragraph(
            "Bird Detection Result:",
            heading_style
        )
    )

    elements.append(Spacer(1, 10))

    image_path = os.path.abspath(result["image"])

    if os.path.exists(image_path):

        pdf_image = PDFImage(
            image_path,
            width=6.2 * inch
        )

        pdf_image._restrictSize(
            6.2 * inch,
            5.5 * inch
        )

        elements.append(pdf_image)

    elements.append(Spacer(1, 20))

    elements.append(
        Paragraph(
            "<b>Identified Bird Type:</b>",
            heading_style
        )
    )

    elements.append(
        Paragraph(
            result["bird"],
            normal_style
        )
    )

    elements.append(Spacer(1, 10))

    elements.append(
        Paragraph(
            "<b>Confidence:</b>",
            heading_style
        )
    )

    elements.append(
        Paragraph(
            f'{result["confidence"]}%',
            normal_style
        )
    )

    elements.append(Spacer(1, 10))

    elements.append(
        Paragraph(
            "<b>Bird Description:</b>",
            heading_style
        )
    )

    elements.append(
        Paragraph(
            result["description"],
            normal_style
        )
    )

    elements.append(Spacer(1, 10))

    elements.append(
        Paragraph(
            "<b>Natural Habitat:</b>",
            heading_style
        )
    )

    elements.append(
        Paragraph(
            result["habitat"],
            normal_style
        )
    )

    doc.build(
    elements,
    onFirstPage=add_pdf_metadata
)

    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="bird_detection_result.pdf",
        mimetype="application/pdf"
    )

# VIEW BIRD DETAILS

@app.route("/bird-details/<bird_type>")
def bird_details(bird_type):

    if "username" not in session:
        return redirect("/login")

    if session["role"] != "admin":
        return redirect("/dashboard")


    cursor.execute("""
        SELECT description, habitat
        FROM birds
        WHERE bird_type=%s
    """, (bird_type,))


    bird = cursor.fetchone()


    if not bird:

        return {
            "description": "",
            "habitat": ""
        }


    return {
        "description": bird[0] or "",
        "habitat": bird[1] or ""
    }

# EDIT BIRD DETAILS

@app.route("/edit-bird/<bird_type>", methods=["POST"])
def edit_bird(bird_type):

    if "username" not in session:
        return {
            "success": False
        }

    if session["role"] != "admin":
        return {
            "success": False
        }


    data = request.get_json()

    description = data.get("description", "")
    habitat = data.get("habitat", "")


    cursor.execute("""
        UPDATE birds
        SET description=%s,
            habitat=%s
        WHERE bird_type=%s
    """, (
        description,
        habitat,
        bird_type
    ))


    db.commit()


    return {
        "success": True
    }

# MANAGE DATASET

@app.route("/manage-dataset")
def manage_dataset():

    if "username" not in session:
        return redirect("/login")

    if session["role"] != "admin":
        return redirect("/dashboard")

    return render_template(
        "manage_dataset.html"
    )

# TRAINING PAGE
@app.route("/training")
def training():
    if "username" not in session:
        return redirect("/login")

    if session["role"] != "admin":
        return redirect("/dashboard")

    status = request.args.get("status")

    return render_template("training.html", status=status)


# TRAIN MODEL
@app.route("/train-model", methods=["POST"])
def train_model_web():
    if "username" not in session:
        return redirect("/login")

    if session["role"] != "admin":
        return redirect("/dashboard")

    try:

        # Train model
        train_model()

        # Reload new trained model
        global model
        model = tf.keras.models.load_model(
            "model/bird_model.keras"
        )

        return redirect("/training?status=success")

    except Exception as e:

        print("Training Error:", e)

        return redirect("/training?status=error")

    
# VIEW DATASET IMAGES

@app.route("/manage-dataset/<bird_type>")
def view_dataset(bird_type):

    if "username" not in session:
        return redirect("/login")

    if session["role"] != "admin":
        return redirect("/dashboard")

    folder = os.path.join("dataset", bird_type)

    if not os.path.exists(folder):
        os.makedirs(folder)

    images = []

    for filename in os.listdir(folder):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            images.append(filename)

    total_images = len(images)

    return render_template(
        "dataset_images.html",
        bird_type=bird_type,
        images=images,
        total_images=total_images
    )
# ADD DATASET IMAGE

@app.route("/add-dataset/<bird_type>", methods=["GET", "POST"])
def add_dataset(bird_type):

    if "username" not in session:
        return redirect("/login")

    if session["role"] != "admin":
        return redirect("/dashboard")

    allowed_birds = [
        "Crow",
        "Duck",
        "Eagle",
        "Owl",
        "Parrot"
    ]

    if bird_type not in allowed_birds:
        return redirect("/manage-dataset")

    if request.method == "POST":

        files = request.files.getlist("images")

        folder = os.path.join(
            "dataset",
            bird_type
        )

        os.makedirs(
            folder,
            exist_ok=True
        )

        for file in files:

            if file and file.filename:

                filepath = os.path.join(
                    folder,
                    file.filename
                )

                file.save(filepath)

        return redirect(
            "/manage-dataset/" + bird_type
        )

    return render_template(
        "add_dataset.html",
        bird_type=bird_type
    )

# DELETE DATASET IMAGE

@app.route("/delete-dataset/<bird_type>/<filename>")
def delete_dataset(bird_type, filename):

    if "username" not in session:
        return redirect("/login")

    if session["role"] != "admin":
        return redirect("/dashboard")

    allowed_birds = [
        "Crow",
        "Duck",
        "Eagle",
        "Owl",
        "Parrot"
    ]

    if bird_type not in allowed_birds:
        return redirect("/manage-dataset")

    filepath = os.path.join(
        "dataset",
        bird_type,
        filename
    )

    if os.path.exists(filepath):
        os.remove(filepath)

    return redirect(
        "/manage-dataset/" + bird_type
    )

# SHOW DATASET IMAGE

@app.route("/dataset/<bird_type>/<filename>")
def dataset_image(bird_type, filename):

    if "username" not in session:
        return redirect("/login")

    if session["role"] != "admin":
        return redirect("/dashboard")

    folder = os.path.join(
        "dataset",
        bird_type
    )

    return send_from_directory(
        folder,
        filename
    )

@app.route("/delete-admin-report/<int:report_id>")
def delete_admin_report(report_id):

    if "username" not in session:
        return redirect("/login")

    if session["role"] != "admin":
        return redirect("/dashboard")

    cursor.execute("""
        DELETE FROM detection_history
        WHERE id=%s
    """, (report_id,))

    db.commit()

    return redirect("/report")

@app.route("/report")
def report():

    if "username" not in session:
        return redirect("/login")

    search = request.args.get("search", "")

    # ADMIN REPORT
    if session["role"] == "admin":

        cursor.execute("""
            SELECT
                detection_history.id,
                detection_history.bird_type,
                detection_history.image,
                detection_history.confidence,
                detection_history.date,
                birds.description,
                birds.habitat
            FROM detection_history
            LEFT JOIN birds
            ON detection_history.bird_type = birds.bird_type
            WHERE
                detection_history.bird_type LIKE %s
                OR detection_history.date LIKE %s
                OR birds.description LIKE %s
                OR birds.habitat LIKE %s
            ORDER BY detection_history.id DESC
        """, (
            "%" + search + "%",
            "%" + search + "%",
            "%" + search + "%",
            "%" + search + "%"
        ))

        records = cursor.fetchall()

        return render_template(
            "admin_report.html",
            records=records,
            search=search
        )

    # USER REPORT
    else:

        cursor.execute("""
            SELECT
                detection_history.id,
                detection_history.image,
                detection_history.bird_type,
                birds.description,
                birds.habitat,
                detection_history.confidence,
                detection_history.date
            FROM detection_history
            LEFT JOIN birds
            ON detection_history.bird_type = birds.bird_type
            WHERE detection_history.username=%s
            AND (
                detection_history.bird_type LIKE %s
                OR detection_history.date LIKE %s
                OR birds.description LIKE %s
                OR birds.habitat LIKE %s
            )
            ORDER BY detection_history.id DESC
        """, (
            session["username"],
            "%" + search + "%",
            "%" + search + "%",
            "%" + search + "%",
            "%" + search + "%"
        ))

        records = cursor.fetchall()

        return render_template(
            "user_report.html",
            records=records,
            search=search
        )

# DOWNLOAD USER REPORT AS PDF
@app.route("/download-report/<int:report_id>")
def download_report(report_id):

    if "username" not in session:
        return redirect("/login")

    cursor.execute("""
        SELECT
            detection_history.image,
            detection_history.bird_type,
            birds.description,
            birds.habitat,
            detection_history.confidence
        FROM detection_history
        LEFT JOIN birds
        ON detection_history.bird_type = birds.bird_type
        WHERE detection_history.id=%s
        AND detection_history.username=%s
    """, (report_id, session["username"]))

    report = cursor.fetchone()

    if not report:
        return redirect("/report")

    # Guna PDF function yang dah ada
    session["last_detection"] = {
        "image": report[0],
        "bird": report[1],
        "description": report[2] or "",
        "habitat": report[3] or "",
        "confidence": round(float(report[4]), 2)
    }

    return redirect("/download-result")

# DOWNLOAD ADMIN REPORT AS PDF

@app.route("/download-admin-report/<int:report_id>")
def download_admin_report(report_id):

    if "username" not in session:
        return redirect("/login")

    if session["role"] != "admin":
        return redirect("/dashboard")

    cursor.execute("""
        SELECT
            detection_history.image,
            detection_history.bird_type,
            birds.description,
            birds.habitat,
            detection_history.confidence
        FROM detection_history
        LEFT JOIN birds
        ON detection_history.bird_type = birds.bird_type
        WHERE detection_history.id=%s
    """, (report_id,))

    report = cursor.fetchone()

    if not report:
        return redirect("/report")

    session["last_detection"] = {
        "image": report[0],
        "bird": report[1],
        "description": report[2] or "",
        "habitat": report[3] or "",
        "confidence": round(float(report[4]), 2)
    }

    return redirect("/download-result")

# DELETE USER REPORT
@app.route("/delete-report/<int:report_id>")
def delete_user_report(report_id):

    if "username" not in session:
        return redirect("/login")

    # Only user can delete their own report
    if session["role"] != "user":
        return redirect("/admin")

    cursor.execute("""
        DELETE FROM detection_history
        WHERE id=%s AND username=%s
    """, (report_id, session["username"]))

    db.commit()

    return redirect("/report")

# ABOUT
@app.route("/about")
def about():

    if "username" not in session:
        return redirect("/login")

    return render_template(
        "about.html"
    )

@app.route("/admin-login", methods=["GET", "POST"])
def admin_login():

    if request.method == "POST":

        username = request.form["username"]
        password = request.form["password"]

        cursor.execute("""
            SELECT username, role
            FROM users
            WHERE username=%s AND password=%s
        """, (username, password))

        user = cursor.fetchone()

        if user:

            if user[1] == "admin":

                session["username"] = user[0]
                session["role"] = "admin"

                return redirect("/admin")

            else:
                return render_template(
                    "admin_login.html",
                    error="This account is not an admin account."
                )

        else:

            return render_template(
                "admin_login.html",
                error="Invalid username or password."
            )

    return render_template("admin_login.html")

# LOGOUT
@app.route("/logout")
def logout():

    session.clear()

    return redirect("/")

# RUN APP
if __name__=="__main__":

    app.run(debug=True)

