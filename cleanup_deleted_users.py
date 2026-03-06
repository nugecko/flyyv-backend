"""
cleanup_deleted_users.py
Run once: dokku run flyyv python cleanup_deleted_users.py
"""
from db import SessionLocal
from models import AppUser, Alert

db = SessionLocal()

emails = [
    'daniel@tm.show',
    'sagib22@gmail.com',
    'optica.mishka@gmail.com',
    'ohad.rtz@gmail.com',
    'akronsound@protonmail.com',
    'plt.kathryn@gmail.com',
]

for email in emails:
    u = db.query(AppUser).filter(AppUser.email == email).first()
    if u:
        deleted_alerts = db.query(Alert).filter(Alert.user_email == email).delete()
        db.delete(u)
        print(f'Deleted: {email} (+ {deleted_alerts} alerts)')
    else:
        print(f'Not found: {email}')

db.commit()
db.close()
print('Done')
