Contributing to Attention Atlas

Thank you for your interest in making this course better. Contributions of all kinds are welcome — from fixing typos to adding new lessons.

How to contribute
Reporting issues

Found a bug, typo, or something confusing? Open an issue.

Be specific:

What lesson or section?
What's wrong? (typo, unclear explanation, broken interaction, etc.)
What would fix it? (if you have an idea)
Small fixes (typos, clarity)
Fork this repo
Edit the HTML file directly (find your section using Ctrl+F)
Test by opening the file in your browser
Submit a pull request with a clear description of the change

The whole course is in one index.html file, so edits are straightforward.

Adding or improving lessons

The course structure is defined in the JavaScript L array near the bottom of index.html. Each lesson has:

javascript
{
  id: "unique-id",
  n: "lesson-number",
  t: "Lesson title",
  tool: "tool-name-or-none",
  idea: `Big idea in HTML`,
  body: `<div class="step">...</div> ...`,
  quiz: [
    {q: "question", o: ["opt1","opt2","opt3"], a: 0, e: "explanation"},
    ...
  ],
  ref: "Reflection prompt"
}

To add a lesson:

Fork the repo
Add a new object to the L array
Write the content following the existing format
If you add an interactive tool, add a function to TOOLS object
Test thoroughly — make sure quizzes work, tools respond, text renders correctly
Submit a PR with a description of what the lesson teaches and why it's valuable
Adding interactive tools

Tools are functions in the TOOLS object. Example template:

javascript
TOOLS.yourtoolname = el => {
  el.innerHTML = `<div>Your HTML here</div>`;
  // Wire up event listeners and interactivity
  el.querySelector('...').addEventListener('input', () => {
    // Update the display
  });
};

Tools should be:

Minimal — focus on one concept
Responsive — instant feedback to user input
Accessible — work with keyboard, not just mouse
Design guidelines

Visual:

The aesthetic is paper-and-ink (warm porcelain page, deep ink, ultramarine and vermilion accents)
Interactive tools are dark glowing panels embedded in the paper
Use the CSS variables defined at the top of the style block

Writing:

Clear, conversational tone
"Understand first, formalize second" — intuition before equations
Avoid jargon; define terms the first time they appear
Explain why something matters, not just what it does

Pedagogy:

Each lesson should stand alone
Build on previous lessons without requiring them
Interactive tools should make the concept tangible
Quizzes should test understanding, not memorization
Code style
Keep it simple — this is one HTML file with no build step
Avoid external dependencies (fonts are loaded from Google Fonts, which is fine)
Comment complex sections
Test across modern browsers (Chrome, Firefox, Safari, Edge)
Pull request process
Describe your change — what problem does it solve or what does it teach?
Link any related issues — if you're fixing a bug, reference it
Include a screenshot or gif if you're adding visual elements
Be patient — we'll review and provide feedback
Questions?

Open an issue with the label question or discussion.

Adopting this for your course?

If you're using this in teaching, please add your course to ADOPTERS.md. It helps other educators discover what's working.

Thank you for helping make transformer education better. ✨
