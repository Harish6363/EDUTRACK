from pymongo import MongoClient
from werkzeug.security import generate_password_hash

client = MongoClient("mongodb://localhost:27017/")
db = client['edutrack_phase1']

# Clear existing data
db.users.delete_many({})
db.attendance.delete_many({})
db.announcements.delete_many({})
db.timetables.delete_many({}) # Clear timetables as well
db.resources.delete_many({}) # Clear resources

# Admin & Faculty
db.users.insert_one({
    "name": "Admin User",
    "email": "admin@edu.com",
    "role": "admin",
    "password": generate_password_hash("admin123")
})
db.users.insert_one({
    "name": "Faculty One",
    "email": "faculty@edu.com",
    "role": "faculty",
    "password": generate_password_hash("faculty123"),
    "subjects": ["Python", "Data Structures", "Java"]  # <-- NEW FIELD
})

# Students: 3 sections (A/B/C) with 5 students each
for sec in ['A','B','C']:
    for i in range(1, 6):
        db.users.insert_one({
            "name": f"student {sec}{i:03d}",
            "email": f"{sec.lower()}{i:03d}@edu.com",
            "role": "student",
            "password": generate_password_hash("student123"),
            "section": sec,
            "roll_no": f"{sec}{i:03d}",
            "total_classes": 0,
            "attended": 0,
            "marks_card": [] 
        })

print("Seeded sample data successfully.")
print("Admin -> admin@edu.com / admin123")
print("Faculty -> faculty@edu.com / faculty123 (Teaches Python, DSA, Java)")
print("Students -> e.g. a001@edu.com / student123")