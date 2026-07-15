"""
Django models for the mHealth application.
"""
import secrets
from datetime import timedelta

from django.db import models
from django.contrib.auth.models import User
from django.contrib.auth.hashers import make_password, check_password
from django.utils import timezone


class UserProfile(models.Model):
    """Extended profile for each user."""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='userprofile')
    form_submitted = models.BooleanField(default=False)

    def __str__(self):
        return f"Profile({self.user.username})"


class EmailOTP(models.Model):
    """
    A short-lived, hashed one-time password sent by email.

    Used for both signup verification and login 2FA. The code itself is never
    stored in clear text; for signup the pending account (username + already
    hashed password) is held here until the code is verified, so no plaintext
    password ever touches the database.
    """
    PURPOSE_SIGNUP = 'signup'
    PURPOSE_LOGIN = 'login'
    PURPOSE_CHOICES = ((PURPOSE_SIGNUP, 'Signup'), (PURPOSE_LOGIN, 'Login 2FA'))

    TTL = timedelta(minutes=10)
    MAX_ATTEMPTS = 5

    email = models.EmailField(db_index=True)
    purpose = models.CharField(max_length=10, choices=PURPOSE_CHOICES)
    code_hash = models.CharField(max_length=255)
    # Pending-signup payload (blank for login OTPs)
    username = models.CharField(max_length=150, blank=True)
    password_hash = models.CharField(max_length=255, blank=True)
    attempts = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [models.Index(fields=['email', 'purpose'])]

    def set_code(self, raw_code: str):
        self.code_hash = make_password(raw_code)

    def check_code(self, raw_code: str) -> bool:
        return check_password(raw_code, self.code_hash)

    def is_expired(self) -> bool:
        return timezone.now() - self.created_at > self.TTL

    def __str__(self):
        return f"OTP({self.purpose}:{self.email})"


class Device(models.Model):
    """A registered wearable device that streams signal data."""
    ONLINE_THRESHOLD = timedelta(seconds=120)

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='devices')
    device_id = models.CharField(max_length=100)
    name = models.CharField(max_length=120, default='Wearable')
    key = models.CharField(max_length=64, unique=True, db_index=True)
    firmware = models.CharField(max_length=50, blank=True)
    battery = models.IntegerField(null=True, blank=True)
    current_session = models.CharField(max_length=200, blank=True)
    last_seen = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'device_id')

    @staticmethod
    def new_key() -> str:
        return secrets.token_hex(24)

    @property
    def is_online(self) -> bool:
        return bool(self.last_seen and timezone.now() - self.last_seen <= self.ONLINE_THRESHOLD)

    def __str__(self):
        return f"{self.name} ({self.device_id})"


class GoogleSheet(models.Model):
    """Saved Google Sheet reference per user."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='google_sheets')
    title = models.CharField(max_length=255, default='Untitled')
    sheet_url = models.URLField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} – {self.title}"


class ExcelFile(models.Model):
    """Uploaded Excel/CSV file."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True, related_name='excel_files')
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='files/')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class UploadedDocument(models.Model):
    """Category metadata for a file uploaded via /api/files/local/upload/.

    The file itself lives on disk under USER_FILES_BASE_DIR/<username>/ (see
    file_management.py) — this row just tags it with a document type so the
    Files page can group/filter medical records (labs, prescriptions,
    medication lists, imaging) instead of showing an undifferentiated list.
    """
    DOC_TYPE_CHOICES = [
        ('lab_report', 'Lab Report'),
        ('prescription', 'Prescription'),
        ('medication', 'Medication List'),
        ('imaging', 'Imaging / Scan'),
        ('sensor_data', 'Sensor / Device Data'),
        ('other', 'Other'),
    ]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='uploaded_documents')
    filename = models.CharField(max_length=255)
    doc_type = models.CharField(max_length=20, choices=DOC_TYPE_CHOICES, default='other')
    uploaded_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'filename')
        indexes = [models.Index(fields=['user', 'doc_type'])]

    def __str__(self):
        return f"UploadedDocument({self.user.username}/{self.filename}: {self.doc_type})"


class ContactMessage(models.Model):
    """Message submitted through the contact form."""
    name = models.CharField(max_length=100)
    email = models.EmailField()
    subject = models.CharField(max_length=200, blank=True)
    message = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Message from {self.name} ({self.timestamp:%Y-%m-%d})"


class AlertFired(models.Model):
    """Persisted record of an alert rule that fired for a specific session."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='alerts_fired')
    rule = models.ForeignKey('AlertRule', on_delete=models.SET_NULL, null=True, related_name='firings')
    signal = models.CharField(max_length=10)
    label = models.CharField(max_length=120)
    operator = models.CharField(max_length=2)
    threshold = models.FloatField()
    actual_mean = models.FloatField()
    owner = models.CharField(max_length=150)
    session = models.CharField(max_length=200)
    fired_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-fired_at']
        indexes = [models.Index(fields=['user', '-fired_at'])]

    def __str__(self):
        return f"AlertFired({self.label} @ {self.owner}/{self.session})"


class SessionAnnotation(models.Model):
    """A timestamped note attached to a specific point in a session."""
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='annotations')
    owner = models.CharField(max_length=150)
    session = models.CharField(max_length=200)
    offset_sec = models.FloatField()
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['offset_sec']
        indexes = [models.Index(fields=['owner', 'session'])]

    def __str__(self):
        return f"Annotation({self.owner}/{self.session}@{self.offset_sec:.1f}s)"


class AlertRule(models.Model):
    """A per-user threshold rule evaluated when a session is loaded."""
    SIGNAL_CHOICES = [
        ('HR', 'Heart Rate (bpm)'),
        ('EDA', 'Skin Conductance (µS)'),
        ('TEMP', 'Skin Temperature (°C)'),
        ('ACC', 'Movement Magnitude (g)'),
    ]
    OP_CHOICES = [('gt', 'Greater than'), ('lt', 'Less than')]

    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='alert_rules')
    signal = models.CharField(max_length=10, choices=SIGNAL_CHOICES)
    operator = models.CharField(max_length=2, choices=OP_CHOICES)
    threshold = models.FloatField()
    label = models.CharField(max_length=120, blank=True)
    enabled = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        op = '>' if self.operator == 'gt' else '<'
        return f"AlertRule({self.user.username}: {self.signal} {op} {self.threshold})"


class ParticipantConsent(models.Model):
    """Health consent form submitted by participants."""
    unique_id = models.AutoField(primary_key=True)
    age = models.IntegerField()
    gender = models.CharField(max_length=10)
    height = models.FloatField()
    weight = models.FloatField()

    respiratory_conditions = models.CharField(max_length=255, blank=True, null=True)
    cardiovascular_conditions = models.CharField(max_length=255, blank=True, null=True)
    cardiovascular_symptoms = models.CharField(max_length=255, blank=True, null=True)
    metabolic_conditions = models.CharField(max_length=255, blank=True, null=True)
    mental_health_conditions = models.CharField(max_length=255, blank=True, null=True)
    stress_level = models.CharField(max_length=50)

    lifestyle_factors = models.CharField(max_length=255, blank=True, null=True)
    sleep_hours = models.CharField(max_length=50)
    sleep_disorders = models.CharField(max_length=255, blank=True, null=True)

    last_medical_checkup = models.CharField(max_length=50)
    health_concerns = models.CharField(max_length=255, blank=True, null=True)
    date_submitted = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Participant {self.unique_id}"
