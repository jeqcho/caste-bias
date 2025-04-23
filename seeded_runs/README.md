# Notes

Running the same seeded runs will give different results. According to OpenAI [here](https://platform.openai.com/docs/api-reference/chat/create):
> This feature is in Beta. If specified, our system will make a best effort to sample deterministically, such that repeated requests with the same seed and parameters should return the same result. Determinism is not guaranteed, and you should refer to the system_fingerprint response parameter to monitor changes in the backend.

This behavior is shown in the three files in this folder. All these runs were run with `seed=1`.