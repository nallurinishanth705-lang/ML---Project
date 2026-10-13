# Matchday — Football Match Result Predictor

A smooth-scrolling, responsive project website for **Multi-Class Match Result Prediction Using Multinomial Logistic Regression**.

**Students:** N. Nishanth (2520030313), N.V. Karthikeya (2520030307)  
**Guide:** Dr. Shaik Asif · CSE

## Run on Windows

1. Install **Node.js LTS** from https://nodejs.org if it is not already installed.
2. Extract this ZIP.
3. Open the extracted `football-match-predictor-website` folder.
4. Double-click `start-windows.bat`.
5. Wait for the server to start, then open **http://localhost:3000**.
6. Keep the terminal window open while you use the website. Press `Ctrl+C` to stop it.

Alternative terminal commands:

```bash
npm install
npm start
```

## What is included?

- `public/` — website page, styling, smooth scrolling and browser interactions
- `server/` — Node.js API and exported Multinomial Logistic Regression model weights
- `server/model.json` — weights fitted from the project's 20,000-row CSV using the same 80/20 stratified split and preprocessing approach
- `package.json` — Node dependencies and start command
- `start-windows.bat` — one-click Windows launcher

## Website sections

- Match prediction form
- Home Win / Draw / Away Win prediction with estimated class probabilities
- Model workflow
- Test-set accuracy, precision, recall and F1 score
- Confusion matrix
- Project credits

## Model information

The included model weights were fitted using the project's historical features: recent points, average goals scored/conceded, historical win rates, country and league. Current-match final goals are not inputs. The displayed test-set results are approximately:

- Accuracy: 49.78%
- Weighted precision: 47.85%
- Weighted recall: 49.78%
- Weighted F1 score: 45.80%

Predictions are estimates for educational demonstration, not guarantees. The model does not currently account for lineups, injuries, or other live match factors.

## Troubleshooting

- **`node` or `npm` is not recognized:** install Node.js LTS, close and reopen VS Code/PowerShell, then try again.
- **Port 3000 is busy:** close another app using port 3000 or run `set PORT=3001` before `npm start`, then visit `http://localhost:3001`.
- **No internet during first launch:** `npm install` needs internet the first time to download Express.
