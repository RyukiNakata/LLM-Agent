import datetime
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

# 必要な権限：読み取り + 書き込み
SCOPES = ["https://www.googleapis.com/auth/calendar"]


def authenticate_google_calendar():
    """
    Google Calendar API にアクセスするための認証処理。
    認証情報を返す。
    """
    creds = None
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)
        with open("token.json", "w") as token:
            token.write(creds.to_json())
    return creds


def get_upcoming_events(n: int = 10) -> str:
    """
    現在から n 件の予定を取得し、文字列で返す。
    """
    try:
        creds = authenticate_google_calendar()
        service = build("calendar", "v3", credentials=creds)

        now = datetime.datetime.utcnow().isoformat() + "Z"
        events_result = service.events().list(
            calendarId="primary",
            timeMin=now,
            maxResults=n,
            singleEvents=True,
            orderBy="startTime",
        ).execute()
        events = events_result.get("items", [])

        if not events:
            return "次の予定はありません。"

        result = "次の予定：\n"
        for event in events:
            start = event["start"].get("dateTime", event["start"].get("date"))
            summary = event.get("summary", "（無題）")
            result += f"{start} - {summary}\n"
        return result

    except HttpError as error:
        return f"予定の取得中にエラーが発生しました: {error}"


def add_event(summary: str, start_time: str, end_time: str) -> str:
    """
    Google Calendar に新しい予定を追加します。
    start_time, end_time は ISO 8601 形式（例: '2025-04-24T15:00:00+09:00'）
    """
    try:
        creds = authenticate_google_calendar()
        service = build("calendar", "v3", credentials=creds)

        event = {
            'summary': summary,
            'start': {'dateTime': start_time, 'timeZone': 'Asia/Tokyo'},
            'end': {'dateTime': end_time, 'timeZone': 'Asia/Tokyo'}
        }

        created_event = service.events().insert(calendarId='primary', body=event).execute()
        return f"✅ 予定 '{summary}' を追加しました: {created_event.get('htmlLink')}"

    except HttpError as error:
        return f"予定の追加中にエラーが発生しました: {error}"


# コマンドラインからテスト可能
if __name__ == "__main__":
    print(get_upcoming_events(5))
    # 予定追加テスト
    # print(add_event("テスト会議", "2025-04-25T10:00:00+09:00", "2025-04-25T11:00:00+09:00"))
