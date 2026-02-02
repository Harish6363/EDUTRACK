import os
import logging
import re
import uuid
import io
import csv
import json
from datetime import date, datetime, timezone, timedelta
from collections import defaultdict
from pathlib import Path
from difflib import SequenceMatcher
import warnings
import requests

# AI, PDF, & Image Imports
import google.generativeai as genai
import PyPDF2
import PIL.Image

# Flask and Extensions
from flask import (
    Flask, redirect, request, send_file, url_for, flash, session, 
    render_template, send_from_directory, Blueprint, jsonify
)
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from bson.objectid import ObjectId

import pandas as pd 
warnings.filterwarnings('ignore')

# ======================================================================
# *** WHATSAPP CLOUD API INTEGRATION ***
# ======================================================================

WHATSAPP_TOKEN ="EAFk3wM4b3X4BP8Ag8LDYVXyPPj0yKw5WrvJpyrKdx5K1RtmZCI1yYLUtENHivO3Tcs100KfR6sVxZA3j9USJMTm3lDzxnVZCKHpaLvb0MgB4uuNezIZCl7RMimPGnycHG8h6hGF3o4gCb9ZB6Gm6A7nOd6gLPbuluPMq9bEeRB1uQmiMepv6ZCrymzGyLZAh4NbIt9JPyfLnaKCvKmif9tJxajRYj0WzFtKVEgvyqwB8BQfkQZDZD"
PHONE_NUMBER_ID ="779394845268218" 

def send_whatsapp_message(to_number, user_name, email, password):
    url = f"https://graph.facebook.com/{PHONE_NUMBER_ID}/messages"
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messaging_product": "whatsapp",
        "to": to_number.strip('+'),
        "type": "text",
        "text":{"body":f"You Login Crentials are \n E-mail id:{email}\n User Name:{user_name}\n Password:{password}"}
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            return True
        return False
    except Exception as e:
        return False

# ======================================================================
# CONFIGURATION
# ======================================================================

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_secret_change_me')
    MONGO_URI = os.environ.get('MONGO_URI', 'mongodb://localhost:27017/edutrack_phase1')
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
    SECTIONS = ['A', 'B', 'C', 'AI']
    # Limit set to 80 MB
    MAX_CONTENT_LENGTH = 80 * 1024 * 1024 

# --- GEMINI AI CONFIGURATION ---
os.environ["GEMINI_API_KEY"] = "GEMINI API KEY" 
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

PERIOD_MAP = {
    "1": "Period 1 (09:00 - 09:55)",
    "2": "Period 2 (10:00 - 10:55)",
    "3": "Period 3 (11:00 - 11:55)",
    "4": "Period 4 (12:00 - 12:55)",
    "5": "Period 5 (14:00 - 14:55)",
    "6": "Period 6 (15:00 - 15:55)",
    "7": "Period 7 (16:00 - 16:55)",
}

app = Flask(__name__)
app.config.from_object(Config)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('edutrack') 

mongo = PyMongo()
mongo.init_app(app)

# Helper: Extract Text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
    except Exception as e:
        logger.error(f"Error reading PDF: {e}")
    return text

@app.errorhandler(413)
def request_entity_too_large(error):
    flash('File is too large! Please upload a file smaller than 80 MB.', 'danger')
    if request.referrer:
        return redirect(request.referrer)
    return redirect(url_for('login'))

# ======================================================================
# ADMIN BLUEPRINT
# ======================================================================

admin_bp = Blueprint('admin', __name__)

def admin_login_required(func):
    def wrapper(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'admin':
            flash('Unauthorized','danger')
            return redirect(url_for('login'))
        return func(*args, **kwargs)
    wrapper.__name__=func.__name__
    return wrapper

@admin_bp.route('/admin')
@admin_login_required
def dashboard():
    total_students = mongo.db.users.count_documents({'role': 'student'})
    total_faculty = mongo.db.users.count_documents({'role': 'faculty'})
    
    pipeline = [
        {'$match': {'role': 'student'}},
        {'$group': {
            '_id': None,
            'total_attended': {'$sum': '$attended'},
            'total_classes': {'$sum': '$total_classes'}
        }}
    ]
    attendance_data = list(mongo.db.users.aggregate(pipeline))
    overall_attendance = 0.0
    if attendance_data and attendance_data[0]['total_classes'] > 0:
        overall_attendance = round((attendance_data[0]['total_attended'] / attendance_data[0]['total_classes']) * 100, 2)

    kpi = {
        'total_students': total_students,
        'total_faculty': total_faculty,
        'overall_attendance': overall_attendance
    }

    activity_feed = []
    attendance_logs = mongo.db.attendance.find().sort('timestamp', -1).limit(10)
    for log in attendance_logs:
        log['activity_type'] = 'Attendance'
        activity_feed.append(log)
        
    resource_logs = mongo.db.resources.find().sort('timestamp', -1).limit(10)
    for log in resource_logs:
        log['activity_type'] = 'Resource'
        activity_feed.append(log)

    announcement_logs = mongo.db.announcements.find().sort('timestamp', -1).limit(20)
    for log in announcement_logs:
        if log.get('post_type') == 'Log':
            if 'Password Changed' in log.get('title', ''):
                log['activity_type'] = 'Security'
            elif 'Profile Updated' in log.get('title', ''):
                log['activity_type'] = 'Profile'
            else:
                log['activity_type'] = 'Marks'
        else:
            log['activity_type'] = 'Announcement'
        activity_feed.append(log)
    
    activity_feed.sort(key=lambda x: x.get('timestamp'), reverse=True)
    activity_feed = activity_feed[:10]

    return render_template('admin_dashboard.html', kpi=kpi, activity_feed=activity_feed)

@admin_bp.route('/admin/users')
@admin_login_required
def manage_users():
    users = list(mongo.db.users.find())
    return render_template('admin_users.html', users=users, sections=Config.SECTIONS)

@admin_bp.route('/admin/add_user',methods=['POST'])
@admin_login_required
def add_user():
    name = request.form.get('name','').strip()
    email = request.form.get('email','').strip().lower()
    role = request.form.get('role')
    password = request.form.get('password')
    section = request.form.get('section','').strip().upper() or None
    roll = request.form.get('roll_no','').strip().upper() or None
    phone = request.form.get('phone','').strip() or None 
    
    subjects_str = request.form.get('subjects', '').strip()
    subjects_list = [s.strip() for s in subjects_str.split(',') if s.strip()]

    if not all([name,email,role,password]):
        flash('Missing fields','warning')
        return redirect(url_for('admin.manage_users'))
    
    if mongo.db.users.find_one({'email':email}):
        flash('User with this email already exists','warning')
        return redirect(url_for('admin.manage_users'))
    
    doc={
        'name':name,'email':email,'role':role,
        'password':generate_password_hash(password),
        'phone': phone, 
        'total_classes':0,'attended':0,'marks_card':[]
    }
    
    if role=='student':
        doc.update({'section':section,'roll_no':roll})
    elif role=='faculty':
        doc.update({'subjects': subjects_list})
    
    mongo.db.users.insert_one(doc)
    
    if phone:
        try:
            clean_phone = phone.replace('+', '').replace('whatsapp:', '')
            send_whatsapp_message(clean_phone, name, email, password)
            flash('User added successfully! Welcome message sent.', 'success')
        except Exception:
            flash('User added, but welcome message failed to send.', 'warning')
    else:
        flash('User added successfully (no phone number provided).', 'success')
    
    return redirect(url_for('admin.manage_users'))

@admin_bp.route('/admin/bulk_upload', methods=['POST'])
@admin_login_required
def bulk_upload():
    file = request.files.get('user_file')
    if not file or not file.filename.endswith('.csv'):
        flash('Please upload a valid .csv file.', 'warning')
        return redirect(url_for('admin.manage_users'))

    try:
        data = file.read().decode('utf-8')
        csv_file = io.StringIO(data)
        reader = csv.DictReader(csv_file)
        users_to_insert = []
        
        for row in reader:
            name = row.get('name')
            email = row.get('email', '').strip().lower()
            password = row.get('password') 
            role = row.get('role')
            phone = row.get('phone','').strip() or None 

            if not all([name, email, password, role]): continue
            if mongo.db.users.find_one({'email': email}): continue

            doc = {
                'name': name.strip(), 'email': email, 'password': generate_password_hash(password), 
                'role': role.strip(), 'phone': phone, 
                'total_classes': 0, 'attended': 0, 'marks_card': []
            }
            if doc['role'] == 'student':
                doc['section'] = row.get('section', '').strip().upper() or None
                doc['roll_no'] = row.get('roll_no', '').strip().upper() or None
            elif doc['role'] == 'faculty':
                subjects_str = row.get('subjects', '').strip()
                subjects_list = [s.strip() for s in subjects_str.split(',') if s.strip()]
                doc['subjects'] = subjects_list

            users_to_insert.append(doc)
            
            if phone:
                try:
                    clean_phone = phone.replace('+', '').replace('whatsapp:', '')
                    send_whatsapp_message(clean_phone, name, email, password)
                except: pass

        if users_to_insert:
            mongo.db.users.insert_many(users_to_insert)
            flash(f'Successfully added {len(users_to_insert)} new users.', 'success')
        else:
            flash('No new users to add.', 'warning')

    except Exception as e:
        flash(f'An error occurred: {e}', 'danger')

    return redirect(url_for('admin.manage_users'))

@admin_bp.route('/admin/edit_user/<user_id>', methods=['POST'])
@admin_login_required
def edit_user(user_id):
    try:
        subjects_str = request.form.get('subjects', '').strip()
        subjects_list = [s.strip() for s in subjects_str.split(',') if s.strip()]
        update_doc = {
            'name': request.form.get('name').strip(),
            'email': request.form.get('email').strip().lower(),
            'role': request.form.get('role'),
            'phone': request.form.get('phone','').strip() or None, 
            'section': request.form.get('section', '').strip().upper() or None,
            'roll_no': request.form.get('roll_no', '').strip().upper() or None,
        }
        if update_doc['role'] == 'faculty': update_doc['subjects'] = subjects_list
        else: update_doc['subjects'] = [] 
        
        new_password = request.form.get('password')
        if new_password: update_doc['password'] = generate_password_hash(new_password)

        mongo.db.users.update_one({'_id': ObjectId(user_id)}, {'$set': update_doc})
        flash('User updated successfully.', 'success')
    except Exception as e:
        flash(f'An error occurred: {e}', 'danger')

    return redirect(url_for('admin.manage_users'))

@admin_bp.route('/admin/remove/<user_id>')
@admin_login_required
def remove_user(user_id):
    mongo.db.users.delete_one({'_id':ObjectId(user_id)})
    flash('User removed','success')
    return redirect(url_for('admin.manage_users'))

@admin_bp.route('/admin/timetable',methods=['GET','POST'])
@admin_login_required
def manage_timetable():
    if request.method=='POST':
        entry={
            'day': request.form.get('day'),
            'period': request.form.get('period'),
            'time_slot': PERIOD_MAP.get(request.form.get('period'), "Unknown"),
            'subject': request.form.get('subject'),
            'section': request.form.get('section'),
            'faculty_email': request.form.get('faculty_email')
        }
        mongo.db.timetables.insert_one(entry)
        flash('Timetable entry added','success')
        return redirect(url_for('admin.manage_timetable'))
    
    faculty_list = list(mongo.db.users.find({'role': 'faculty'}))
    timetable_entries = list(mongo.db.timetables.find())
    timetable_by_day = defaultdict(list)
    for entry in timetable_entries:
        entry['period_num'] = int(entry.get('period', 99))
        timetable_by_day[entry['day']].append(entry)
    
    return render_template('admin_timetable.html', faculty_list=faculty_list, sections=Config.SECTIONS, timetable_by_day=timetable_by_day, period_map=PERIOD_MAP)

@admin_bp.route('/admin/timetable/delete/<entry_id>')
@admin_login_required
def delete_timetable_entry(entry_id):
    mongo.db.timetables.delete_one({'_id':ObjectId(entry_id)})
    flash('Deleted','success')
    return redirect(url_for('admin.manage_timetable'))

# --- RESET ATTENDANCE ROUTE ---
@admin_bp.route('/admin/reset_attendance')
@admin_login_required
def reset_attendance():
    try:
        # 1. Delete all documents in attendance collection
        mongo.db.attendance.delete_many({})
        
        # 2. Reset counters for all students
        mongo.db.users.update_many(
            {'role': 'student'},
            {'$set': {'total_classes': 0, 'attended': 0}}
        )
        flash('⚠️ System Reset: All attendance data has been deleted and student stats reset to 0.', 'warning')
    except Exception as e:
        flash(f'Error resetting data: {str(e)}', 'danger')
    
    return redirect(url_for('admin.dashboard'))

@admin_bp.route('/announcement_image/<filename>')
def get_announcement_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ======================================================================
# FACULTY BLUEPRINT
# ======================================================================

faculty_bp = Blueprint('faculty', __name__)

def faculty_login_required(func):
    def wrapper(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'faculty':
            flash('Unauthorized', 'danger'); return redirect(url_for('login'))
        return func(*args, **kwargs)
    wrapper.__name__=func.__name__
    return wrapper

@faculty_bp.route('/faculty')
@faculty_login_required
def dashboard():
    current_faculty_email = session.get('email')
    faculty_user = mongo.db.users.find_one({'_id': ObjectId(session['user_id'])})
    faculty_subjects = faculty_user.get('subjects', [])
    sections_taught = list(mongo.db.timetables.distinct('section', {'faculty_email': current_faculty_email}))
    students = list(mongo.db.users.find({'role': 'student', 'section': {'$in': sections_taught}}))
    
    all_attendance = list(mongo.db.attendance.find({'section': {'$in': sections_taught}}))
    attendance_by_roll = defaultdict(list)
    for doc in all_attendance:
        for rec in doc.get('records', []):
            # Safe Date Formatting (Handles "Day 1" as well as "2023-11-01")
            raw_date = doc['date']
            try:
                fmt_date = datetime.fromisoformat(raw_date).strftime('%d-%b-%Y')
            except ValueError:
                fmt_date = raw_date # Fallback to "Day 1"

            attendance_by_roll[rec.get('roll_no')].append({
                'date': fmt_date,
                'status': rec.get('status'), 'subject': doc.get('subject', 'Unknown')
            })

    total_attended_all, total_classes_all = 0, 0
    for s in students:
        t,a = s.get('total_classes',0), s.get('attended',0)
        s['percent'] = round((a/t)*100,2) if t > 0 else 0.0
        s['fine'] = int(max(0,85-s['percent'])*50) if s['percent']<85 else 0
        s['attendance_chart_data'] = json.dumps({'present': a, 'absent': t - a})
        s['attendance_history_list'] = [rec for rec in attendance_by_roll.get(s.get('roll_no'), []) if rec.get('subject') in faculty_subjects]
        total_attended_all += a; total_classes_all += t

    attendance_history = list(mongo.db.attendance.find({'uploaded_by': current_faculty_email}).sort('timestamp', -1).limit(10))
    for att in attendance_history:
        # Safe Date Formatting for History Table
        try:
            att['formatted_date'] = datetime.fromisoformat(att['date']).strftime('%d-%b-%Y')
        except ValueError:
            att['formatted_date'] = att['date'] # Use "Day 1" directly
    
    announcements=list(mongo.db.announcements.find({'post_type': {'$ne': 'Log'}}).sort('timestamp',-1))
    timetable_by_day=defaultdict(list)
    for entry in mongo.db.timetables.find({'faculty_email': current_faculty_email}):
        entry['period_num'] = int(entry.get('period', 99))
        timetable_by_day[entry['day']].append(entry)
        
    kpi_data={'total_students': len(students), 'avg_attendance': round((total_attended_all / total_classes_all) * 100, 2) if total_classes_all > 0 else 0}
    faculty_avg_chart_data = {'present': total_attended_all, 'absent': total_classes_all - total_attended_all}
    insights = [{'type': 'success', 'text': "Positive Trend: Attendance improved by 5%."}]
    
    return render_template('faculty_dashboard.html', students=students, latest=attendance_history, sections=Config.SECTIONS, announcements=announcements, timetable_by_day=timetable_by_day, kpi_data=kpi_data, ai_insights=insights, faculty_avg_chart_data=faculty_avg_chart_data)

@faculty_bp.route('/faculty/attendance', methods=['GET','POST'])
@faculty_login_required
def attendance():
    if request.method=='POST':
        section = request.form.get('section')
        subject = request.form.get('subject')
        uploaded = request.files.get('attendance_sheet')
        manual = request.form.get('absent_list','').strip()

        if uploaded and uploaded.filename:
            try:
                df = pd.read_excel(uploaded) if uploaded.filename.endswith(('.xls', '.xlsx')) else pd.read_csv(uploaded)
                date_cols = [col for col in df.columns if col != 'Roll_Number' and re.match(r'\d{4}-\d{2}-\d{2}', str(col).split(" ")[0])]
                
                if date_cols:
                    for _, row in df.iterrows():
                        roll = str(row['Roll_Number']).strip().upper()
                        for d in date_cols:
                            status = "Present" if str(row[d]).strip().upper() == 'P' else "Absent"
                            date_iso = str(d).split(" ")[0]
                            
                            mongo.db.attendance.update_one(
                                {'date': date_iso, 'section': section, 'subject': subject},
                                {'$addToSet': {'records': {'roll_no': roll, 'status': status}}, 
                                 '$set': {'uploaded_by': session.get('email'), 'timestamp': datetime.now(timezone.utc)}},
                                upsert=True
                            )
                            mongo.db.users.update_one({'roll_no': roll}, {'$inc': {'total_classes': 1, 'attended': 1 if status=='Present' else 0}})
                    flash('Batch update successful.', 'success')
                else:
                    flash('No valid date columns found.', 'warning')
            except Exception as e: flash(f'Error: {e}', 'danger')
        
        elif manual:
            absent_set = {t.strip().upper() for t in re.split(r'[\s,;|]+', manual) if t.strip()}
            all_rolls = {s['roll_no'] for s in mongo.db.users.find({'role':'student','section':section}, {'roll_no':1}) if s.get('roll_no')}
            present_set = all_rolls - absent_set
            
            records = [{'roll_no':rn,'status':'Present'} for rn in present_set] + [{'roll_no':rn,'status':'Absent'} for rn in absent_set]
            mongo.db.attendance.update_one(
                {'date': date.today().isoformat(), 'section':section, 'subject':subject},
                {'$set':{'records':records, 'uploaded_by':session.get('email'), 'timestamp':datetime.now(timezone.utc)}}, upsert=True
            )
            mongo.db.users.update_many({'roll_no': {'$in': list(all_rolls)}}, {'$inc':{'total_classes':1}})
            mongo.db.users.update_many({'roll_no':{'$in':list(present_set)}}, {'$inc':{'attended':1}})
            flash('Attendance recorded.', 'success')

        return redirect(url_for('faculty.dashboard'))
    
    faculty_user = mongo.db.users.find_one({'_id': ObjectId(session['user_id'])})
    return render_template('faculty_attendance.html', sections=Config.SECTIONS, faculty_subjects=faculty_user.get('subjects', []))

# --- NEW: SMART ATTENDANCE SCANNER (WITH SECTION VALIDATION + OVERWRITE) ---
@faculty_bp.route('/faculty/attendance/smart_upload', methods=['POST'])
@faculty_login_required
def smart_attendance_upload():
    if 'attendance_image' not in request.files:
        flash('No image uploaded', 'warning')
        return redirect(url_for('faculty.attendance'))
        
    file = request.files['attendance_image']
    section = request.form.get('section')
    subject = request.form.get('subject')
    
    if not file or not section or not subject:
        flash('Please select Section, Subject, and an Image.', 'warning')
        return redirect(url_for('faculty.attendance'))

    try:
        img = PIL.Image.open(file)
        model = genai.GenerativeModel('gemini-2.0-flash')
        
        # --- PROMPT ---
        prompt = """
        Analyze this attendance sheet image. It may contain records for ONE day or MULTIPLE days (columns).
        
        1. Identify all DATE columns. Instead of specific dates (like Nov 1), Label them sequentially as "Day 1", "Day 2", "Day 3", etc. from left to right.
        2. Identify all STUDENT rows (Roll Numbers).
        3. Extract the status for EVERY student for EVERY column.
        
        Return a strictly valid JSON list of objects. 
        Format: [{"date": "Day 1", "roll_no": "Student_ID", "status": "Present/Absent"}]
        
        Rules:
        - Convert symbols (✓, P,  .) to 'Present'.
        - Convert symbols (X, A, blank) to 'Absent'.
        """
        
        response = model.generate_content([prompt, img])
        
        # Clean & Parse
        clean_json = response.text.replace('```json', '').replace('```', '').strip()
        start, end = clean_json.find('['), clean_json.rfind(']') + 1
        if start != -1 and end != -1:
            records = json.loads(clean_json[start:end])
        else:
            records = json.loads(clean_json)

        # ============================================================
        # 🛡️ SECURITY CHECK: VALIDATE SECTION ROSTER
        # ============================================================
        
        # 1. Get all actual roll numbers for this section from DB
        valid_students = mongo.db.users.find({'section': section, 'role': 'student'})
        valid_roll_numbers = set(s['roll_no'].strip().upper() for s in valid_students if s.get('roll_no'))
        
        if not valid_roll_numbers:
            flash(f"Error: No students found in database for Section {section}. Cannot validate upload.", 'danger')
            return redirect(url_for('faculty.attendance'))

        # 2. Extract unique roll numbers found by AI in the image
        scanned_roll_numbers = set(str(r.get('roll_no', '')).strip().upper() for r in records)
        
        # 3. Count how many scanned rolls actually exist in this section
        # We use set intersection to find matches
        matching_rolls = scanned_roll_numbers.intersection(valid_roll_numbers)
        match_count = len(matching_rolls)
        total_scanned = len(scanned_roll_numbers) if len(scanned_roll_numbers) > 0 else 1
        
        # 4. Calculate Match Percentage
        # If less than 30% of scanned names belong to this section, it's likely the wrong sheet.
        match_percentage = (match_count / total_scanned) * 100
        
        print(f"DEBUG: Section {section} Validation - Matches: {match_count}/{total_scanned} ({match_percentage:.2f}%)")

        if match_percentage < 30:
            flash(f"⚠️ Upload Rejected: This sheet does not look like Section {section}. Only {int(match_percentage)}% of roll numbers matched. Did you upload the wrong file?", "danger")
            return redirect(url_for('faculty.attendance'))

        # ============================================================
        # END VALIDATION - PROCEED TO UPDATE
        # ============================================================

        # Group by Date label
        records_by_date = defaultdict(list)
        for rec in records:
            date_str = rec.get('date', 'Day 1') 
            records_by_date[date_str].append(rec)

        count_dates_updated = 0

        # Process each date group independently
        for date_key, day_records in records_by_date.items():
            
            # 1. REVERT OLD DATA STATS (Prevent Double Counting)
            old_entry = mongo.db.attendance.find_one({'date': date_key, 'section': section, 'subject': subject})
            if old_entry:
                for old_rec in old_entry.get('records', []):
                    u_roll = old_rec.get('roll_no')
                    u_status = old_rec.get('status')
                    inc_total = -1
                    inc_attended = -1 if u_status == 'Present' else 0
                    mongo.db.users.update_one({'roll_no': u_roll}, {'$inc': {'total_classes': inc_total, 'attended': inc_attended}})

            # 2. Prepare New Data & Apply NEW Stats
            new_db_records = []
            for rec in day_records:
                r_roll = str(rec.get('roll_no', '')).strip().upper()
                
                # SKIP INVALID STUDENTS (Double check)
                if r_roll not in valid_roll_numbers:
                    continue 

                r_status = rec.get('status', 'Absent').capitalize()
                if r_status not in ['Present', 'Absent']: r_status = 'Absent'
                
                new_db_records.append({'roll_no': r_roll, 'status': r_status})

                # Add new stats
                inc_total = 1
                inc_attended = 1 if r_status == 'Present' else 0
                mongo.db.users.update_one({'roll_no': r_roll}, {'$inc': {'total_classes': inc_total, 'attended': inc_attended}})

            # 3. OVERWRITE Attendance Log
            mongo.db.attendance.update_one(
                {'date': date_key, 'section': section, 'subject': subject},
                {'$set': {
                    'records': new_db_records, 
                    'uploaded_by': session.get('email'), 
                    'timestamp': datetime.now(timezone.utc)
                }},
                upsert=True
            )
            count_dates_updated += 1

        flash(f'Smart Scan Complete! Verified and processed data for {count_dates_updated} days.', 'success')

    except Exception as e:
        print(f"AI OCR ERROR: {e}")
        flash(f'Error processing image: {str(e)}', 'danger')

    return redirect(url_for('faculty.attendance'))

@faculty_bp.route('/faculty/announcement', methods=['GET','POST'])
@faculty_login_required
def post_announcement():
    if request.method=='POST':
        image = request.files.get('image_file')
        filename = None
        if image and image.filename:
            filename = f"{uuid.uuid4().hex}_{secure_filename(image.filename)}"
            image.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        mongo.db.announcements.insert_one({
            'title': request.form.get('title'), 'content': request.form.get('content'),
            'post_type': request.form.get('post_type'), 'posted_by': session.get('email'),
            'timestamp': datetime.now(timezone.utc), 'image_filename': filename
        })
        flash('Announcement posted','success')
        return redirect(url_for('faculty.dashboard'))
    return render_template('post_announcement.html')

@faculty_bp.route('/faculty/resources', methods=['GET', 'POST'])
@faculty_login_required
def post_resource():
    faculty_user = mongo.db.users.find_one({'_id': ObjectId(session['user_id'])})
    if request.method == 'POST':
        file = request.files.get('resource_file')
        if file and file.filename:
            filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            mongo.db.resources.insert_one({
                'title': request.form.get('title'), 'subject': request.form.get('subject'),
                'section': request.form.get('section'), 'filename': filename,
                'original_filename': secure_filename(file.filename), 'posted_by': session.get('email'),
                'timestamp': datetime.now(timezone.utc)
            })
            flash('Resource uploaded.', 'success')
            return redirect(url_for('faculty.dashboard'))
    return render_template('post_resource.html', faculty_subjects=faculty_user.get('subjects', []), sections=Config.SECTIONS)

@faculty_bp.route('/faculty/marks',methods=['GET','POST'])
@faculty_login_required
def upload_marks():
    if request.method=='POST':
        student_id = request.form.get('student_id')
        semester = int(request.form.get('semester'))
        marks=[]
        for i in range(1,6):
            sub = request.form.get(f'subject_{i}')
            scr = request.form.get(f'score_{i}')
            if sub and scr: marks.append({'subject':sub, 'score':int(scr)})
        
        if marks:
            mongo.db.users.update_one(
                {'_id': ObjectId(student_id)}, 
                {'$push': {'marks_card': {'semester': semester, 'marks': marks}}}
            )
            mongo.db.announcements.insert_one({
                'title': 'Uploaded Marks', 'content': f'Marks uploaded for Sem {semester}',
                'post_type': 'Log', 'posted_by': session.get('email'), 'timestamp': datetime.now(timezone.utc)
            })
            flash('Marks saved.', 'success')
            return redirect(url_for('faculty.dashboard'))
            
    return render_template('faculty_marks.html', students=list(mongo.db.users.find({'role':'student'})))

@faculty_bp.route('/faculty/bulk_marks', methods=['POST'])
@faculty_login_required
def bulk_upload_marks():
    try:
        semester = int(request.form.get('semester'))
        file = request.files.get('marks_file')
        df = pd.read_excel(file) if file.filename.endswith(('.xls','.xlsx')) else pd.read_csv(file)
        
        count = 0
        for _, row in df.iterrows():
            roll = str(row['Roll_Number']).strip().upper()
            student = mongo.db.users.find_one({'roll_no': roll, 'role': 'student'})
            if not student: continue
            
            marks = []
            for i in range(1, 6):
                if f'Subject_{i}' in row and f'Score_{i}' in row and pd.notna(row[f'Score_{i}']):
                    marks.append({'subject': str(row[f'Subject_{i}']), 'score': int(row[f'Score_{i}'])})
            
            if marks:
                mongo.db.users.update_one({'_id': student['_id']}, {'$push': {'marks_card': {'semester': semester, 'marks': marks}}})
                count += 1
        flash(f'Bulk upload complete. Updated {count} students.', 'success')
    except Exception as e: flash(f'Error: {e}', 'danger')
    return redirect(url_for('faculty.dashboard'))

@faculty_bp.route('/announcement_image/<filename>')
@faculty_login_required
def get_announcement_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ======================================================================
# STUDENT BLUEPRINT
# ======================================================================

student_bp = Blueprint('student', __name__)

def student_login_required(func):
    def wrapper(*args, **kwargs):
        if 'user_id' not in session or session.get('role') != 'student':
            flash('Unauthorized', 'danger'); return redirect(url_for('login'))
        return func(*args, **kwargs)
    wrapper.__name__=func.__name__
    return wrapper

@student_bp.route('/student')
@student_login_required
def dashboard():
    user = mongo.db.users.find_one({'_id':ObjectId(session['user_id'])})
    total,attended=user.get('total_classes',0),user.get('attended',0)
    percent=round((attended/total)*100,2) if total > 0 else 0
    
    faculty_map = {f['email']: f['name'] for f in mongo.db.users.find({'role': 'faculty'})}
    timetable_by_day=defaultdict(list)
    subjects_in_section = set()
    for entry in mongo.db.timetables.find({'section': user.get('section')}):
        entry['faculty_name'] = faculty_map.get(entry.get('faculty_email'), 'N/A')
        entry['period_num'] = int(entry.get('period', 99))
        timetable_by_day[entry['day']].append(entry)
        subjects_in_section.add(entry.get('subject'))

    attendance_history = []
    subject_attendance = defaultdict(lambda: {'attended': 0, 'total': 0})
    for doc in mongo.db.attendance.find({'section': user.get('section')}):
        for rec in doc.get('records',[]):
            if rec.get('roll_no')==user.get('roll_no'):
                # Safe Date Formatting for Student
                raw_date = doc['date']
                try:
                    fmt_date = datetime.fromisoformat(raw_date).strftime('%d-%b-%Y')
                except ValueError:
                    fmt_date = raw_date # Use "Day 1" directly
                
                attendance_history.append({'date': fmt_date, 'status': rec.get('status'), 'subject': doc.get('subject')})
                subject_attendance[doc.get('subject')]['total'] += 1
                if rec.get('status') == 'Present': subject_attendance[doc.get('subject')]['attended'] += 1
    
    subject_attendance_data = [{'subject': k, 'attended': v['attended'], 'total': v['total'], 'percent': round((v['attended']/v['total'])*100,2) if v['total']>0 else 0} for k,v in subject_attendance.items()]
    
    today_status = 'Pending'
    today_log = mongo.db.attendance.find_one({'date': date.today().isoformat(), 'section': user.get('section')})
    if today_log:
        for rec in today_log.get('records',[]):
            if rec.get('roll_no') == user.get('roll_no'): today_status = rec.get('status'); break

    return render_template(
        'student_dashboard.html', user=user, percent=percent, 
        attendance_history=sorted(attendance_history, key=lambda x: x['date'], reverse=True),
        subject_attendance_data=subject_attendance_data,
        announcements=list(mongo.db.announcements.find({'post_type': {'$ne': 'Log'}}).sort('timestamp', -1)),
        timetable_by_day=timetable_by_day, today_status=today_status,
        attendance_chart_data=json.dumps({'present': attended, 'absent': total - attended}),
        marks_card_data=user.get('marks_card', []),
        resources=list(mongo.db.resources.find({'subject': {'$in': list(subjects_in_section)}, 'section': user.get('section')}).sort('timestamp', -1))
    )

# --- NEW: ADDED MISSING DOWNLOAD & RESOURCE ROUTES FOR STUDENT ---

@student_bp.route('/student/download/timetable')
@student_login_required
def download_timetable():
    user = mongo.db.users.find_one({'_id': ObjectId(session['user_id'])})
    timetable = list(mongo.db.timetables.find({'section': user.get('section')}))
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Day', 'Period', 'Time', 'Subject', 'Faculty Email'])
    for t in timetable:
        writer.writerow([t.get('day'), t.get('period'), PERIOD_MAP.get(t.get('period')), t.get('subject'), t.get('faculty_email')])
    
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode('utf-8')), mimetype='text/csv', as_attachment=True, download_name='timetable.csv')

@student_bp.route('/student/download/marks')
@student_login_required
def download_marks():
    user = mongo.db.users.find_one({'_id': ObjectId(session['user_id'])})
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Semester', 'Subject', 'Score'])
    
    for sem in user.get('marks_card', []):
        for m in sem.get('marks', []):
            writer.writerow([sem.get('semester'), m.get('subject'), m.get('score')])
            
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode('utf-8')), mimetype='text/csv', as_attachment=True, download_name='marks_report.csv')

@student_bp.route('/student/download/attendance')
@student_login_required
def download_attendance():
    user = mongo.db.users.find_one({'_id': ObjectId(session['user_id'])})
    
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['Date', 'Subject', 'Status'])
    
    for doc in mongo.db.attendance.find({'section': user.get('section')}):
        for rec in doc.get('records',[]):
            if rec.get('roll_no')==user.get('roll_no'):
                writer.writerow([doc.get('date'), doc.get('subject'), rec.get('status')])
                
    output.seek(0)
    return send_file(io.BytesIO(output.getvalue().encode('utf-8')), mimetype='text/csv', as_attachment=True, download_name='attendance_report.csv')

@student_bp.route('/student/resource/<filename>')
@student_login_required
def get_resource(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@student_bp.route('/student/announcement_image/<filename>')
@student_login_required
def get_announcement_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- END NEW ROUTES ---

@student_bp.route('/student/edit_profile', methods=['GET', 'POST'])
@student_login_required
def edit_profile():
    user_id = session.get('user_id')
    if request.method == 'POST':
        name = request.form.get('name')
        phone = request.form.get('phone')
        
        mongo.db.users.update_one(
            {'_id': ObjectId(user_id)},
            {'$set': {'name': name, 'phone': phone}}
        )
        
        # Log update
        mongo.db.announcements.insert_one({
            'title': 'Profile Updated',
            'content': f'{name} updated their profile details.',
            'post_type': 'Log',
            'posted_by': session.get('email'),
            'timestamp': datetime.now(timezone.utc)
        })
        
        flash('Profile updated successfully.', 'success')
        return redirect(url_for('student.dashboard'))
        
    user = mongo.db.users.find_one({'_id': ObjectId(user_id)})
    return render_template('student_edit_profile.html', user=user)

@student_bp.route('/student/change_password', methods=['GET', 'POST'])
@student_login_required
def change_password():
    if request.method == 'POST':
        old_pass = request.form.get('old_password')
        new_pass = request.form.get('new_password')
        confirm_pass = request.form.get('confirm_password')
        
        user = mongo.db.users.find_one({'_id': ObjectId(session.get('user_id'))})
        
        if not check_password_hash(user['password'], old_pass):
            flash('Incorrect old password.', 'danger')
            return redirect(url_for('student.change_password'))
            
        if new_pass != confirm_pass:
            flash('New passwords do not match.', 'danger')
            return redirect(url_for('student.change_password'))
            
        mongo.db.users.update_one(
            {'_id': ObjectId(session.get('user_id'))},
            {'$set': {'password': generate_password_hash(new_pass)}}
        )
        
        mongo.db.announcements.insert_one({
            'title': 'Password Changed', 
            'content': f"User {user.get('email')} changed their password.",
            'post_type': 'Log', 
            'posted_by': session.get('email'), 
            'timestamp': datetime.now(timezone.utc)
        })
        
        flash('Password changed successfully.', 'success')
        return redirect(url_for('student.dashboard'))
        
    return render_template('change_password.html')

# --- TASK MANAGER ROUTES ---

@student_bp.route('/student/tasks', methods=['GET', 'POST'])
@student_login_required
def tasks():
    user_id = session.get('user_id')
    
    if request.method == 'POST':
        title = request.form.get('title')
        subject = request.form.get('subject')
        deadline = request.form.get('deadline') 
        priority = request.form.get('priority')
        
        if not all([title, subject, deadline]):
            flash('Title, Subject, and Deadline are required.', 'warning')
            return redirect(url_for('student.tasks'))
            
        task_doc = {
            'user_id': user_id,
            'title': title,
            'subject': subject,
            'deadline': deadline,
            'priority': priority,
            'status': 'Pending',
            'created_at': datetime.now(timezone.utc)
        }
        
        mongo.db.tasks.insert_one(task_doc)
        flash('Task added to your study plan!', 'success')
        return redirect(url_for('student.tasks'))
        
    pending_tasks = list(mongo.db.tasks.find({'user_id': user_id, 'status': 'Pending'}).sort('deadline', 1))
    completed_tasks = list(mongo.db.tasks.find({'user_id': user_id, 'status': 'Completed'}).sort('deadline', -1).limit(5))
    
    user = mongo.db.users.find_one({'_id': ObjectId(user_id)})
    timetable_entries = list(mongo.db.timetables.find({'section': user.get('section')}))
    subjects = sorted(list(set(t['subject'] for t in timetable_entries)))
    
    return render_template('student_tasks.html', pending_tasks=pending_tasks, completed_tasks=completed_tasks, subjects=subjects)

@student_bp.route('/student/task/complete/<task_id>')
@student_login_required
def complete_task(task_id):
    mongo.db.tasks.update_one(
        {'_id': ObjectId(task_id), 'user_id': session.get('user_id')},
        {'$set': {'status': 'Completed'}}
    )
    flash('Great job! Task marked as completed.', 'success')
    return redirect(url_for('student.tasks'))

@student_bp.route('/student/task/delete/<task_id>')
@student_login_required
def delete_task(task_id):
    mongo.db.tasks.delete_one({'_id': ObjectId(task_id), 'user_id': session.get('user_id')})
    flash('Task removed.', 'info')
    return redirect(url_for('student.tasks'))

# --- NEW: AI STUDY TOOLS ROUTES (WITH DELETE) ---

@student_bp.route('/student/study_tools', methods=['GET', 'POST'])
@student_login_required
def study_tools():
    user_id = session.get('user_id')
    
    # 1. Handle File Upload & Flashcard Generation
    if request.method == 'POST':
        if 'study_file' not in request.files:
            flash('No file part', 'warning')
            return redirect(request.url)
            
        file = request.files['study_file']
        if file.filename == '':
            flash('No selected file', 'warning')
            return redirect(request.url)
            
        if file:
            try:
                filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # A. Extract Text
                extracted_text = extract_text_from_pdf(file_path)
                
                # --- DEBUGGING: Check if PDF was read correctly ---
                print(f"DEBUG: Extracted {len(extracted_text)} characters.")
                
                if not extracted_text or len(extracted_text.strip()) < 50:
                    flash('Error: The PDF appears empty. If this is a SCANNED document (image), the system cannot read it. Please upload a text-based PDF.', 'danger')
                    return redirect(request.url)
                    
                # B. Generate Flashcards using Gemini
                # USING YOUR VERIFIED AVAILABLE MODEL: gemini-2.0-flash
                model = genai.GenerativeModel('gemini-2.0-flash')
                
                prompt = (
                    f"Analyze the following study notes and generate 5 study flashcards (Question and Answer) "
                    f"and 1 summary of the topic. Return the response as strictly valid JSON in this format: "
                    f"{{ \"summary\": \"...\", \"flashcards\": [{{\"question\": \"...\", \"answer\": \"...\"}}] }} \n\n"
                    f"Study Notes Content: {extracted_text[:15000]}" # Limit text
                )
                
                response = model.generate_content(prompt)
                
                # Check if response was blocked
                if not response.parts:
                    print("DEBUG: Safety filters blocked the response.")
                    flash('AI Error: Content was blocked by safety filters. Try a different document.', 'warning')
                    return redirect(request.url)

                # Clean response
                clean_json = response.text.replace('```json', '').replace('```', '').strip()
                
                # Try parsing
                try:
                    ai_data = json.loads(clean_json)
                except json.JSONDecodeError:
                    # Fallback cleanup if AI adds extra text
                    start = clean_json.find('{')
                    end = clean_json.rfind('}') + 1
                    if start != -1 and end != -1:
                        ai_data = json.loads(clean_json[start:end])
                    else:
                        raise ValueError("AI response was not valid JSON.")

                # C. Save to DB
                mongo.db.study_materials.insert_one({
                    'user_id': user_id,
                    'filename': filename,
                    'original_name': file.filename,
                    'extracted_text': extracted_text, 
                    'summary': ai_data.get('summary'),
                    'flashcards': ai_data.get('flashcards'),
                    'created_at': datetime.now(timezone.utc)
                })
                flash('AI successfully generated flashcards from your notes!', 'success')

            except Exception as e:
                # This prints the actual error to your terminal so you can see it
                print(f"CRITICAL ERROR: {e}")
                flash(f'System Error: {str(e)}', 'danger')
                
            return redirect(url_for('student.study_tools'))

    # Fetch user's study materials
    materials = list(mongo.db.study_materials.find({'user_id': user_id}).sort('created_at', -1))
    return render_template('student_study_tools.html', materials=materials)

@student_bp.route('/student/study_tools/delete/<material_id>')
@student_login_required
def delete_study_material(material_id):
    try:
        # Find the material to ensure it belongs to the user
        material = mongo.db.study_materials.find_one({
            '_id': ObjectId(material_id),
            'user_id': session.get('user_id')
        })

        if material:
            # 1. Remove the file from the server
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], material['filename'])
            if os.path.exists(file_path):
                os.remove(file_path)

            # 2. Delete from Database
            mongo.db.study_materials.delete_one({'_id': ObjectId(material_id)})
            flash('Study material deleted successfully.', 'success')
        else:
            flash('Material not found or unauthorized.', 'danger')

    except Exception as e:
        flash(f'Error deleting material: {str(e)}', 'danger')

    return redirect(url_for('student.study_tools'))

@student_bp.route('/student/chat_with_notes', methods=['POST'])
@student_login_required
def chat_with_notes():
    data = request.json
    material_id = data.get('material_id')
    user_question = data.get('question')
    
    material = mongo.db.study_materials.find_one({'_id': ObjectId(material_id)})
    if not material:
        return {'answer': 'Error: Notes not found.'}, 404
        
    # Simple RAG: Provide context + question to Gemini
    context = material.get('extracted_text', '')[:15000] # Provide mostly full context
    
    try:
        # USING YOUR VERIFIED AVAILABLE MODEL: gemini-2.0-flash
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = (
            f"You are a helpful AI tutor. Answer the student's question based ONLY on the provided notes below.\n"
            f"If the answer is not in the notes, say so.\n\n"
            f"--- NOTES START ---\n{context}\n--- NOTES END ---\n\n"
            f"Student Question: {user_question}"
        )
        response = model.generate_content(prompt)
        return {'answer': response.text}
    except Exception as e:
        return {'answer': f"AI Error: {str(e)}"}, 500

# --- END AI ROUTES ---

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email')
        # Here you would typically generate an OTP and email it
        # For this prototype, we will simulate it
        session['reset_email'] = email # Store email to use in reset step
        flash(f'OTP sent to {email} (Simulation)', 'info')
        return redirect(url_for('reset_password'))
    return render_template('forgot_password.html')

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'POST':
        otp = request.form.get('otp')
        new_password = request.form.get('new_password')
        confirm_password = request.form.get('confirm_password')
        
        if new_password != confirm_password:
            flash('Passwords do not match!', 'danger')
            return redirect(url_for('reset_password'))
        
        # Logic to verify OTP would go here.
        # Logic to update password in MongoDB:
        email = session.get('reset_email')
        if email:
            hashed_pw = generate_password_hash(new_password)
            # Try finding in all roles
            mongo.db.users.update_one({'email': email}, {'$set': {'password': hashed_pw}})
            session.pop('reset_email', None)
            flash('Password reset successfully. Please login.', 'success')
            return redirect(url_for('login'))
        else:
            flash('Session expired. Please try forgot password again.', 'danger')
            return redirect(url_for('forgot_password'))

    return render_template('reset_password.html')

@app.route('/')
def index():
    if 'user_id' in session: return redirect(url_for(f"{session.get('role')}.dashboard"))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        role = request.form.get('role')
        
        # 1. Normalize Inputs
        email = email.strip().lower()
        # Force role to lowercase for consistent URL building
        role_slug = role.lower() 
        
        # 2. Find User
        # Note: We check 'role' in DB (which might be capitalized like 'Faculty')
        # But we use role_slug for the URL redirect
        user = mongo.db.users.find_one({'email': email})
        
        if user and user.get('role').lower() == role_slug:
            if check_password_hash(user['password'], password):
                session['user_id'] = str(user['_id'])
                session['role'] = role_slug
                session['email'] = user.get('email')
                flash('Logged in successfully', 'success')
                return redirect(url_for(f'{role_slug}.dashboard'))
        
        flash('Invalid credentials or role mismatch', 'danger')
    
    return render_template('login.html')

@app.route('/logout')
def logout(): 
    session.clear()
    return redirect(url_for('login'))

# ======================================================================
# REGISTER BLUEPRINTS
# ======================================================================
app.register_blueprint(admin_bp)
app.register_blueprint(faculty_bp)
app.register_blueprint(student_bp)

if __name__ == '__main__':

    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
