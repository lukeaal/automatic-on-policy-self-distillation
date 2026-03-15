"""
Student model setup backed by Hugging Face Hub.
The student model is the model that will be evaluated and self-distilled.
"""

from pathlib import Path

from huggingface_hub import snapshot_download


class StudentModel:
    """Represents a student model downloaded from Hugging Face."""

    def __init__(self, student_model_id: str) -> None:
        self.student_model_id = student_model_id
        self.local_path: Path | None = None

    def setup(self) -> Path:
        """Download the student model snapshot and cache its local path."""
        model_dir = snapshot_download(repo_id=self.student_model_id)
        self.local_path = Path(model_dir)
        return self.local_path
