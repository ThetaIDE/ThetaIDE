def test_launch_button_hands_off_to_user_terminal(monkeypatch):
    from frontend.app import Window
    from types import SimpleNamespace
    calls = []
    monkeypatch.setattr('frontend.site_launch.launch', calls.append)
    host = SimpleNamespace(composed_config={'site': {'scheduler': 'batch'}})
    Window.launch_training(host)
    assert calls == [host]
