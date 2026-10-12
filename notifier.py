"""Stand-in for the private Telegram notifier: prints instead of sending."""


def send_text(text: str) -> None:
    print(text)
