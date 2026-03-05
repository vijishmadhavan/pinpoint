"""Tests for api/memory.py — conversation, memory, settings, reminders."""


class TestConversation:
    def test_save_and_retrieve(self, client):
        # Save two messages
        r = client.post("/conversation", json={"session_id": "test-session", "role": "user", "content": "Hello"})
        assert r.status_code == 200
        assert r.json()["success"] is True

        r = client.post("/conversation", json={"session_id": "test-session", "role": "assistant", "content": "Hi there!"})
        assert r.status_code == 200

        # Retrieve history
        r = client.get("/conversation/history", params={"session_id": "test-session"})
        assert r.status_code == 200
        data = r.json()
        assert len(data["messages"]) == 2
        assert data["messages"][0]["content"] == "Hello"
        assert data["messages"][1]["content"] == "Hi there!"
        assert data["message_count"] == 2

    def test_empty_history(self, client):
        r = client.get("/conversation/history", params={"session_id": "nonexistent"})
        assert r.status_code == 200
        assert r.json()["messages"] == []
        assert r.json()["message_count"] == 0

    def test_reset(self, client):
        client.post("/conversation", json={"session_id": "s1", "role": "user", "content": "msg1"})
        client.post("/conversation", json={"session_id": "s1", "role": "user", "content": "msg2"})
        r = client.post("/conversation/reset", json={"session_id": "s1"})
        assert r.status_code == 200
        assert r.json()["deleted_count"] == 2
        # Verify empty
        r = client.get("/conversation/history", params={"session_id": "s1"})
        assert r.json()["messages"] == []

    def test_search(self, client):
        client.post("/conversation", json={"session_id": "s2", "role": "user", "content": "Tell me about pizza"})
        client.post("/conversation", json={"session_id": "s2", "role": "assistant", "content": "Pizza is delicious"})
        r = client.get("/conversation/search", params={"q": "pizza"})
        assert r.status_code == 200
        assert r.json()["count"] == 2

    def test_invalid_role(self, client):
        r = client.post("/conversation", json={"session_id": "s1", "role": "system", "content": "test"})
        assert r.status_code == 400

    def test_empty_content(self, client):
        r = client.post("/conversation", json={"session_id": "s1", "role": "user", "content": "  "})
        assert r.status_code == 400


class TestMemory:
    def test_save_and_search(self, client):
        r = client.post("/memory", json={"fact": "I love hiking in the mountains", "category": "preferences"})
        assert r.status_code == 200
        assert r.json()["success"] is True
        assert "id" in r.json()

        r = client.get("/memory/search", params={"q": "hiking mountains"})
        assert r.status_code == 200
        assert r.json()["count"] >= 1

    def test_list_memories(self, client):
        client.post("/memory", json={"fact": "Likes coffee", "category": "preferences"})
        client.post("/memory", json={"fact": "Lives in NYC", "category": "places"})

        r = client.get("/memory/list")
        assert r.status_code == 200
        assert r.json()["count"] >= 2

        # Filter by category
        r = client.get("/memory/list", params={"category": "preferences"})
        assert r.status_code == 200
        for m in r.json()["memories"]:
            assert m["category"] == "preferences"

    def test_delete_memory(self, client):
        r = client.post("/memory", json={"fact": "Temporary fact", "category": "general"})
        mid = r.json()["id"]
        r = client.delete(f"/memory/{mid}")
        assert r.status_code == 200
        assert r.json()["deleted_id"] == mid
        # Verify deleted
        r = client.delete(f"/memory/{mid}")
        assert r.status_code == 404

    def test_forget_memory(self, client):
        client.post("/memory", json={"fact": "My favorite color is blue", "category": "preferences"})
        r = client.post("/memory/forget", json={"description": "favorite color blue"})
        assert r.status_code == 200
        assert r.json()["success"] is True

    def test_memory_context(self, client):
        client.post("/memory", json={"fact": "Prefers dark mode", "category": "preferences"})
        client.post("/memory", json={"fact": "Friend named John", "category": "people"})
        r = client.get("/memory/context")
        assert r.status_code == 200
        data = r.json()
        assert data["count"] >= 2
        assert "Prefers dark mode" in data["text"]

    def test_empty_fact_rejected(self, client):
        r = client.post("/memory", json={"fact": "", "category": "general"})
        assert r.status_code == 400


class TestSettings:
    def test_set_and_get(self, client):
        r = client.post("/setting", params={"key": "theme", "value": "dark"})
        assert r.status_code == 200
        r = client.get("/setting", params={"key": "theme"})
        assert r.status_code == 200
        assert r.json()["value"] == "dark"

    def test_get_nonexistent(self, client):
        r = client.get("/setting", params={"key": "nonexistent"})
        assert r.status_code == 200
        assert r.json()["value"] is None

    def test_overwrite(self, client):
        client.post("/setting", params={"key": "lang", "value": "en"})
        client.post("/setting", params={"key": "lang", "value": "fr"})
        r = client.get("/setting", params={"key": "lang"})
        assert r.json()["value"] == "fr"


class TestReminders:
    def test_create_and_list(self, client):
        r = client.post("/reminders", json={
            "chat_jid": "123@s.whatsapp.net",
            "message": "Take medicine",
            "trigger_at": "2026-03-05T09:00:00"
        })
        assert r.status_code == 200
        rid = r.json()["id"]

        r = client.get("/reminders", params={"chat_jid": "123@s.whatsapp.net"})
        assert r.status_code == 200
        assert len(r.json()["reminders"]) >= 1

    def test_delete_reminder(self, client):
        r = client.post("/reminders", json={
            "chat_jid": "456@s.whatsapp.net",
            "message": "Meeting",
            "trigger_at": "2026-03-06T10:00:00"
        })
        rid = r.json()["id"]
        r = client.delete(f"/reminders/{rid}")
        assert r.status_code == 200

    def test_update_reminder(self, client):
        r = client.post("/reminders", json={
            "chat_jid": "789@s.whatsapp.net",
            "message": "Call dentist",
            "trigger_at": "2026-03-07T14:00:00"
        })
        rid = r.json()["id"]
        r = client.put(f"/reminders/{rid}", params={"trigger_at": "2026-03-08T14:00:00"})
        assert r.status_code == 200
        assert r.json()["trigger_at"] == "2026-03-08T14:00:00"
