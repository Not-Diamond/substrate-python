import marimo

__generated_with = "0.3.12"
app = marimo.App(width="medium")


@app.cell
def __():
    import os
    import json
    import base64
    import marimo as mo
    from substrate import Substrate, GenerateJSON, GenerateText, RunPython, sb

    api_key = os.environ.get("SUBSTRATE_API_KEY")
    api_key = api_key or "YOUR_API_KEY"
    mo.md(f"`{api_key}`")
    return (
        GenerateJSON,
        GenerateText,
        RunPython,
        Substrate,
        api_key,
        base64,
        json,
        mo,
        os,
        sb,
    )


@app.cell
def __(Substrate, api_key):
    substrate = Substrate(
        api_key=api_key,
        backend="v1",
    )
    return (substrate,)


@app.cell
def __(mo):
    question = mo.ui.text(
        placeholder="Question",
        value="What is the 88th fibonacci number?",
        full_width=True,
    ).form()
    question
    return (question,)


@app.cell
def __(GenerateText, RunPython, question, sb):
    prompt = f"""
    {question.value}

    Think step by step and return Python code to solve the problem. The Python runtime does not have network or filesystem access, but does include the entire standard library. Read input from stdin and write output to stdout. Your code should end by printing the answer as json along with an explanation using json.dumps. Wrap the code in your response inside a <code></code> tag. Don't forget to import json.
    """
    gen_code = GenerateText(
        prompt=prompt,
        node="Llama3Instruct70B",
    )
    run_code = RunPython(
        code=sb.jq(gen_code.future.text, 'ascii_downcase | split("<code>") | .[1] | split("</code>") | .[0]')
    )
    return gen_code, prompt, run_code


@app.cell
def __(gen_code, mo, run_code, substrate):
    res = substrate.run(gen_code, run_code)
    viz = substrate.visualize(gen_code, run_code)
    mo.md(f"[visualize]({viz})")
    return res, viz


@app.cell
def __(json, res):
    print(json.dumps(res.json, indent=2))
    # mo.tree(res.json)
    return


@app.cell
def __(mo, viz):
    mo.md(f"[visualize]({viz})")
    return


if __name__ == "__main__":
    app.run()
