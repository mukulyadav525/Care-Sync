"""
URL Configuration for mHealth API backend.

All routes are prefixed with /api/ for clarity.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

from rest_framework_simplejwt.views import TokenRefreshView

# Import view modules
from login.views import auth as auth_views
from login.views import pages as pages_views
from login.views import file_management as fm_views
from login.views import visualization as viz_views
from login.views import hl7 as hl7_views
from login.views import device as device_views
from login.views import devices as devices_views
from login.views import annotations as annotation_views
from login.views import alerts as alert_views
from login.views import report as report_views

# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------
auth_urls = [
    path('signup/', auth_views.signup, name='api-signup'),
    path('send-otp/', auth_views.send_otp, name='api-send-otp'),
    path('verify-otp/', auth_views.verify_otp, name='api-verify-otp'),
    path('login/', auth_views.login_view, name='api-login'),
    path('login/verify/', auth_views.verify_login_otp, name='api-login-verify'),
    path('logout/', auth_views.logout_view, name='api-logout'),
    path('change-password/', auth_views.change_password, name='api-change-password'),
    path('me/', auth_views.me, name='api-me'),
    path('token/refresh/', TokenRefreshView.as_view(), name='api-token-refresh'),
]

# ---------------------------------------------------------------------------
# Connected-device registry endpoints
# ---------------------------------------------------------------------------
device_registry_urls = [
    path('', devices_views.list_devices, name='api-devices'),
    path('register/', devices_views.register_device, name='api-device-register'),
    path('heartbeat/', devices_views.heartbeat, name='api-device-heartbeat'),
    path('<int:pk>/delete/', devices_views.delete_device, name='api-device-delete'),
]

# ---------------------------------------------------------------------------
# File management endpoints
# ---------------------------------------------------------------------------
file_urls = [
    # Local files
    path('local/', fm_views.local_files_view, name='api-local-files'),
    path('local/upload/', fm_views.upload_local_file, name='api-local-upload'),
    path('local/<str:username>/<str:filename>/', fm_views.view_file_data, name='api-file-data'),
    path('local/<str:username>/<str:filename>/delete/', fm_views.delete_local_file, name='api-delete-file'),
    # Google Sheets
    path('sheets/', fm_views.view_google_sheets, name='api-sheets-list'),
    path('sheets/upload/', fm_views.upload_google_sheet, name='api-sheets-upload'),
    path('sheets/<int:file_id>/', fm_views.google_sheet_detail, name='api-sheet-detail'),
    path('sheets/<int:file_id>/delete/', fm_views.delete_google_sheet, name='api-sheet-delete'),
    # Misc
    path('form-submit/', fm_views.form_submit, name='api-form-submit'),
]

# ---------------------------------------------------------------------------
# Visualization endpoints
# ---------------------------------------------------------------------------
viz_urls = [
    # Local file visualizations
    path('local-ppg/<str:filename>/', viz_views.view_local_ppg, name='api-local-ppg'),
    path('local-gsr/<str:filename>/', viz_views.view_local_gsr, name='api-local-gsr'),
    # Google Sheet PPG
    path('google-ppg/<int:file_id>/', viz_views.display_csv, name='api-google-ppg'),
    # PPG graph from ExcelFile
    path('ppg-graph/<int:file_id>/', viz_views.generate_ppg_graph, name='api-ppg-graph'),
    # Actigraphy
    path('actigraphy/<str:filename>/', viz_views.homme, name='api-actigraphy'),
    path('actigraphy/<str:username>/<str:filename>/', viz_views.homme, name='api-actigraphy-user'),
    path('actigraphy/<str:filename>/day/<int:day_id>/', viz_views.actigraphy_day_view, name='api-actigraphy-day'),
    path('actigraphy/<str:username>/<str:filename>/day/<int:day_id>/', viz_views.actigraphy_day_view, name='api-actigraphy-day-user'),
    path('actigraphy-stats/<str:filename>/', viz_views.actigraphy_stats, name='api-actigraphy-stats'),
    path('actigraphy-weekly/<str:filename>/', viz_views.actigraphy_weekly, name='api-actigraphy-weekly'),
    path('actigraphy-weekly/<str:username>/<str:filename>/', viz_views.actigraphy_weekly, name='api-actigraphy-weekly-user'),
    path('compact/<str:filename>/', viz_views.compact, name='api-compact'),
]

# ---------------------------------------------------------------------------
# HL7 endpoints
# ---------------------------------------------------------------------------
hl7_urls = [
    path('generate/', hl7_views.generate_hl7, name='api-hl7-generate'),
    path('download/', hl7_views.download_hl7, name='api-hl7-download'),
    path('view/<int:file_id>/', hl7_views.view_hl7, name='api-hl7-view'),
    path('local/<str:filename>/', hl7_views.convert_local_csv_to_hl7, name='api-hl7-local'),
    path('pdf/<str:filename>/', hl7_views.download_hl7_pdf, name='api-hl7-pdf'),
]

# ---------------------------------------------------------------------------
# Device signal portal endpoints
# ---------------------------------------------------------------------------
device_urls = [
    path('sessions/', device_views.list_sessions, name='api-device-sessions'),
    path('sessions/trends/', device_views.session_trends, name='api-device-trends'),
    path('sessions/compare/', device_views.compare_sessions, name='api-device-compare'),
    path('sessions/<str:owner>/<str:name>/', device_views.session_detail, name='api-device-session-detail'),
    path('sessions/<str:owner>/<str:name>/annotations/', annotation_views.annotation_list, name='api-annotations'),
    path('sessions/<str:owner>/<str:name>/report.pdf', report_views.session_report_pdf, name='api-session-report'),
    path('annotations/<int:pk>/', annotation_views.annotation_delete, name='api-annotation-delete'),
]

alert_urls = [
    path('', alert_views.alert_list, name='api-alerts'),
    path('history/', alert_views.alert_history, name='api-alert-history'),
    path('history/clear/', alert_views.clear_alert_history, name='api-alert-history-clear'),
    path('<int:pk>/', alert_views.alert_detail, name='api-alert-detail'),
]

# ---------------------------------------------------------------------------
# Root URL config
# ---------------------------------------------------------------------------
urlpatterns = [
    # Django admin
    path('admin/', admin.site.urls),
    # API namespaces
    path('api/health/', pages_views.health_check, name='api-health'),
    path('api/profile/', pages_views.profile, name='api-profile'),
    path('api/contact/', pages_views.contact_view, name='api-contact'),
    path('api/admin/patients/', pages_views.admin_patients, name='api-admin-patients'),
    path('api/auth/', include(auth_urls)),
    path('api/files/', include(file_urls)),
    path('api/visualization/', include(viz_urls)),
    path('api/hl7/', include(hl7_urls)),
    path('api/device/', include(device_urls)),
    path('api/devices/', include(device_registry_urls)),
    path('api/alerts/', include(alert_urls)),
]

# Serve media files in development
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
