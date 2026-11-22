// Python (Pyodide) runs here, in a web worker: the page does not freeze while the model answers,
// and examiner.post_json() may use a synchronous XMLHttpRequest, which browsers allow in workers.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.3/full/pyodide.js");

const FILES = ["examiner.py", "data/topics.json", "data/students.json"];

const ready = (async () => {
  const py = await loadPyodide();
  py.FS.mkdirTree("/home/pyodide/data");
  for (const name of FILES) {
    const text = await (await fetch(name)).text();       // same folder as this script
    py.FS.writeFile("/home/pyodide/" + name, text);
  }
  py.runPython(`
import json, sys
sys.path.insert(0, "/home/pyodide")
import examiner

session = examiner.ExamSession()

def state(error=None):
    return json.dumps({"history": session.history, "record": session.record, "error": error},
                      ensure_ascii=False)

def new_exam():
    global session
    session = examiner.ExamSession()
    return state()

def send(text, key, model):
    try:
        if text:
            session.reply(text, key, model)
        else:                       # empty text: finish a turn that was cut by an error
            session.continue_turn(key, model)
        return state()
    except examiner.LLMError as e:
        return state(str(e))
`);
  return py;
})();

onmessage = async (event) => {
  try {
    const py = await ready;
    const { type, text, key, model } = event.data;
    const out = type === "new" ? py.globals.get("new_exam")() : py.globals.get("send")(text, key, model);
    postMessage(JSON.parse(out));
  } catch (err) {
    postMessage({ error: String(err) });
  }
};
