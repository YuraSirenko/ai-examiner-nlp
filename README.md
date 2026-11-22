# AI-екзаменатор з курсу NLP

LLM-агент (Groq, tool calling), який приймає іспит з курсу «Обробка природної мови».
Python-код агента (`examiner.py`) виконується прямо в браузері через Pyodide (`worker.js`),
тому ключ Groq API надсилається лише до Groq і ніде не зберігається.

Інструменти агента: `start_exam(email, name)`, `get_next_topic()`, `end_exam(email, score, history)`.
Дані: `data/topics.json` (теми курсу), `data/students.json` (вигадані студенти).

Демо-студент: Сіренко Юрій, `yurii.sirenko@example.com`.
