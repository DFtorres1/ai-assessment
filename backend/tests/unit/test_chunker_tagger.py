from __future__ import annotations

from knowledge.chunker import Chunk, chunk_document
from knowledge.tagger import tag_chunks


class TestChunkDocument:
    def test_empty_text_returns_empty_list(self):
        assert chunk_document("", doc_name="doc", page=1) == []

    def test_short_text_produces_one_chunk(self):
        chunks = chunk_document(
            "Reset your password via the forgot password link.",
            doc_name="Login Guide",
            page=3,
            section="Password Reset",
        )
        assert len(chunks) == 1
        chunk = chunks[0]
        assert "forgot password" in chunk.text.lower()
        assert chunk.metadata["doc_name"] == "Login Guide"
        assert chunk.metadata["page"] == 3
        assert chunk.metadata["section"] == "Password Reset"

    def test_chunk_has_unique_id(self):
        chunks1 = chunk_document("Some text.", doc_name="Doc A", page=1)
        chunks2 = chunk_document("Some text.", doc_name="Doc B", page=1)
        assert chunks1[0].id != chunks2[0].id

    def test_long_text_produces_multiple_chunks(self):
        long_text = (
            "The account lockout policy locks your account after five failed attempts. " * 50
        )
        chunks = chunk_document(long_text, doc_name="Policy", page=1)
        assert len(chunks) >= 2

    def test_metadata_tags_initialized_empty(self):
        chunks = chunk_document("Some banking text.", doc_name="Guide", page=0)
        assert "tags" in chunks[0].metadata
        assert chunks[0].metadata["tags"] == ""

    def test_whitespace_only_text_returns_empty(self):
        assert chunk_document("   \n\t  ", doc_name="doc", page=0) == []


class TestTagChunks:
    def _make_chunk(self, text: str) -> Chunk:
        return Chunk(id="test", text=text, metadata={"tags": ""})

    def test_password_reset_tags(self):
        chunk = self._make_chunk("To reset your password, use the forgot password link.")
        result = tag_chunks([chunk])
        tags = result[0].metadata["tags"].split(",")
        assert "password" in tags
        assert "reset" in tags

    def test_lockout_tags(self):
        chunk = self._make_chunk("Your account is locked after five failed login attempts.")
        result = tag_chunks([chunk])
        tags = result[0].metadata["tags"].split(",")
        assert "lockout" in tags

    def test_unlock_tag(self):
        chunk = self._make_chunk("Contact support to unlock your account.")
        result = tag_chunks([chunk])
        tags = result[0].metadata["tags"].split(",")
        assert "unlock" in tags

    def test_mfa_tags(self):
        chunk = self._make_chunk("Enable MFA for multi-factor authentication with your 2FA app.")
        result = tag_chunks([chunk])
        tags = result[0].metadata["tags"].split(",")
        assert "mfa" in tags

    def test_verification_tag(self):
        chunk = self._make_chunk("Enter the verification code sent to your email.")
        result = tag_chunks([chunk])
        tags = result[0].metadata["tags"].split(",")
        assert "verification" in tags

    def test_remember_device_tag(self):
        chunk = self._make_chunk("Check 'remember this device' to skip MFA on trusted devices.")
        result = tag_chunks([chunk])
        tags = result[0].metadata["tags"].split(",")
        assert "remember_me" in tags

    def test_username_tag(self):
        chunk = self._make_chunk("Recover your username via the forgot username link.")
        result = tag_chunks([chunk])
        tags = result[0].metadata["tags"].split(",")
        assert "username" in tags

    def test_phone_banking_tag(self):
        chunk = self._make_chunk("Phone banking users can call the IVR system.")
        result = tag_chunks([chunk])
        tags = result[0].metadata["tags"].split(",")
        assert "phone_banking" in tags

    def test_no_matching_keywords_empty_tags(self):
        chunk = self._make_chunk("Interest rates vary by account type.")
        result = tag_chunks([chunk])
        assert result[0].metadata["tags"] == ""

    def test_modifies_chunks_in_place(self):
        chunks = [self._make_chunk("Reset your password."), self._make_chunk("No keywords here.")]
        result = tag_chunks(chunks)
        assert result is chunks
        assert "password" in result[0].metadata["tags"]
        assert result[1].metadata["tags"] == ""

    def test_duplicate_tags_not_added(self):
        chunk = self._make_chunk("Reset your password. Password reset link.")
        result = tag_chunks([chunk])
        tags = result[0].metadata["tags"].split(",")
        assert tags.count("password") == 1
        assert tags.count("reset") == 1
