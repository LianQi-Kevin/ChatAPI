# ChatAPI

This repository is the flask api backend, 
Used to encapsulate the LLM model and provide a call API

---

### Model Supported (In plan)

* [ChatGLM-6B](https://github.com/THUDM/ChatGLM-6B)
* [ChatGLM2-6B](https://github.com/THUDM/ChatGLM2-6B)

---

### Branch

*   ChatGLM2-6B
*   使用RTX3090，测试一次对话的时间（单次对话时间随Token数增加而延长）

|                | Float16 | Int4   | Int8   |
|----------------|---------|--------|--------|
| Use fastllm    | 4.29s   | 2.9s   | 3.3s   |
| Unused fastllm | 9.79s   | 14.76s | 18.90s |

