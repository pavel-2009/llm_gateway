"""Locust workload for exercising the gateway with the mock Ollama service."""

from random import choice

from locust import HttpUser, between, task


_PROMPTS = (
    "Summarize a short fictional story.",
    "Suggest a name for a small garden.",
    "Explain why leaves change color.",
    "Write a friendly one-sentence greeting.",
)


class GatewayUser(HttpUser):
    wait_time = between(0.2, 1.0)

    @task
    def generate(self) -> None:
        with self.client.post(
            "/generate",
            json=self._payload(),
            name="/generate",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"gateway returned HTTP {response.status_code}")
                return

            try:
                payload = response.json()
            except ValueError:
                response.failure("gateway returned invalid JSON")
                return

            if not isinstance(payload, dict) or not payload.get("text"):
                response.failure("gateway response is missing generated text")

    @task
    def extract(self) -> None:
        with self.client.post(
            "/extract",
            json=self._payload(),
            name="/extract",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"gateway returned HTTP {response.status_code}")
                return

            try:
                payload = response.json()
            except ValueError:
                response.failure("gateway returned invalid JSON")
                return

            if (
                not isinstance(payload, dict)
                or payload.get("sentiment")
                not in {"positive", "negative", "mixed", "neutral"}
                or not isinstance(payload.get("topics"), list)
                or not isinstance(payload.get("rating"), int)
            ):
                response.failure("gateway returned invalid extracted data")

    @task
    def stream(self) -> None:
        with self.client.post(
            "/stream",
            json=self._payload(),
            name="/stream",
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"gateway returned HTTP {response.status_code}")
            elif "event: done" not in response.text:
                response.failure("gateway stream did not complete successfully")

    @staticmethod
    def _payload() -> dict[str, str]:
        return {"prompt": choice(_PROMPTS), "model": "light"}
