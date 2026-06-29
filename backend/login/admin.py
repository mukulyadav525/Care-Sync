"""
Django Admin registrations for the login app.
"""
from django.contrib import admin
from django.http import HttpResponseForbidden

from .models import (
    ExcelFile, ContactMessage, GoogleSheet, UserProfile,
    ParticipantConsent, Device
)


@admin.register(ExcelFile)
class ExcelFileAdmin(admin.ModelAdmin):
    list_display = ('name', 'user', 'uploaded_at')
    fields = ['name', 'file', 'user']

    def save_model(self, request, obj, form, change):
        if request.user.is_superuser:
            super().save_model(request, obj, form, change)
        else:
            return HttpResponseForbidden("You are not allowed to upload files.")

    def has_delete_permission(self, request, obj=None):
        return request.user.is_superuser


@admin.register(ContactMessage)
class ContactMessageAdmin(admin.ModelAdmin):
    list_display = ('name', 'email', 'subject', 'timestamp')
    list_filter = ('timestamp',)
    search_fields = ('name', 'email', 'subject')


@admin.register(GoogleSheet)
class GoogleSheetAdmin(admin.ModelAdmin):
    list_display = ('user', 'title', 'sheet_url', 'created_at')
    search_fields = ('user__username', 'title')


@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'form_submitted')
    list_filter = ('form_submitted',)
    search_fields = ('user__username',)


@admin.register(ParticipantConsent)
class ParticipantConsentAdmin(admin.ModelAdmin):
    list_display = ('unique_id', 'age', 'gender', 'date_submitted')


@admin.register(Device)
class DeviceAdmin(admin.ModelAdmin):
    list_display = ('name', 'device_id', 'user', 'battery', 'last_seen', 'is_online')
    list_filter = ('user',)
    search_fields = ('name', 'device_id', 'user__username')
    readonly_fields = ('key', 'last_seen', 'created_at')

    @admin.display(boolean=True, description='Online')
    def is_online(self, obj):
        return obj.is_online
