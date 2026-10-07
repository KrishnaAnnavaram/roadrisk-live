# The writing standard: ASD-STE100 Simplified Technical English

Use these rules for every README and for `docs/ste-style-guide.md` in each repository. Copy this file
into the repository as `docs/ste-style-guide.md` and add a **project vocabulary** section (Section 3)
with the technical names and technical verbs of that project.

## 1. The writing rules

### Words

1. Use one word for one meaning, and one meaning for one word. Do not use synonyms for variety.
2. Use a word only as one part of speech. For example, `test` is a noun or a verb, `check` is a verb.
3. Do not use phrasal verbs (`set up`, `carry out`, `find out`, `pick up`, `look up`, `come up with`).
   Use one verb: `prepare`, `do`, `find`, `get`, `make`.
4. Do not use an `-ing` form as a noun or an adjective (`the running job`, `after indexing`).
   Exception: a technical name, a file name, a command or a status value.
5. Do not use contractions (`don't`, `it's`, `can't`). Do not use slang or idioms
   (`out of the box`, `under the hood`, `at a glance`, `gotcha`, `bells and whistles`).
6. Do not use `and/or`. Write `A, B or both`.
7. Do not use `should`, `could`, `would` or `may` for instructions. Use `must` for a rule, the
   imperative for a step and `can` for a possibility.
8. Keep the articles `a`, `an` and `the` in sentences.
9. Do not make a noun cluster of more than three words. A technical name is one word.

### Sentences

1. A procedural sentence (an instruction) has a maximum of **20 words**.
2. A descriptive sentence has a maximum of **25 words**.
3. Write one instruction in one sentence.
4. Use the imperative for an instruction: `Run the tests.` Not `The tests should be run.`
5. Use the active voice. Use the passive voice only when the agent of the action is not important.
6. Use only the simple present, the simple past and the simple future.
7. Put a condition before the instruction: `If the index is stale, build it again.`
8. Do not use semicolons in sentences. Write two sentences.

### Paragraphs, notes and warnings

1. A paragraph has one topic and a maximum of **6 sentences**. Start with the topic sentence.
2. A warning or a caution starts with a clear command. Then it gives the reason.
3. A note gives information. It does not give an instruction.
4. Use a vertical list for a sequence or a set of conditions. Each item of a numbered procedure is one step.

### Tables, headings and diagrams

1. A table cell can be a short phrase. If a cell has a sentence, the sentence obeys the rules.
2. A heading is a noun phrase (`The cost model`) or an imperative (`Run the demo`).
   Do not start a heading with an `-ing` form.
3. A diagram label is a short phrase. Use the same terms as the text.

### What STE does not change

Code, commands, file names, paths, field names, environment variables, status values, enum values,
product names and URLs stay exactly as they are. They are technical names. Put them in backticks.

## 2. General words to replace

| Do not use | Use |
|---|---|
| utilize, leverage | use |
| in order to | to |
| set up | prepare, install, configure |
| carry out, perform | do |
| make sure, ensure | make sure (allowed), or `check that` |
| a lot of, lots of | many, much |
| e.g., i.e. | for example, that is |
| should (instruction) | must (rule) / imperative (step) |
| might, may (possibility) | can |
| very, really, just, simply, easily | (delete) |
| seamless, robust, powerful, blazing | (delete or give a measured fact) |

## 3. Project vocabulary

These terms have one meaning in the roadrisk-live documentation. Code names are in backticks.

### 3.1 Technical names (nouns)

| Term | Meaning | Do not use |
|---|---|---|
| **accident** | One row of US-Accidents: one reported traffic accident. | crash (for a row), incident, event |
| **severity** | The US-Accidents impact level 1 to 4 (traffic delay, not injuries). | risk, danger level |
| **source** | The data provider column of US-Accidents (`Source1`, `Source2`, ...). | feed, origin |
| **feature table** | The validated table with the model input columns and the target. | dataset (alone), frame |
| **split** | `train`, `valid` or `test`, by date. | fold, partition |
| **training window** | The dates of the train split, up to `ROADRISK_TRAIN_END`. | period, era |
| **weather vocabulary** | The eleven weather categories in `VOCAB`. | weather labels, classes |
| **weather category** | One value of the weather vocabulary. | condition (for the category) |
| **observation** | One parsed live weather response in training units. | snapshot, reading |
| **provider** | A source of observations: `OpenWeatherClient` or `FixtureWeatherProvider`. | API (alone), service |
| **fixture** | A recorded example weather response in `serving/fixtures`. | mock, sample file |
| **geo cell** | A 0.5-degree latitude and longitude square, for example `44.5_-93.5`. | H3 cell, tile, region |
| **pipeline** | The scikit-learn object with preprocessing and classifier. | model (alone), workflow |
| **bundle** | `severity_model.joblib`: the pipeline, the feature list, the classes and the window. | artifact, pickle |
| **live score** | The severity probabilities for one observation. | prediction (alone), forecast |
| **daily count** | The number of accidents on one calendar day. | volume, traffic |
| **forecaster** | A daily count model: `naive`, `seasonal_naive`, `holt_winters`, `lag_regression`. | predictor, LSTM |
| **origin** | The last day of history before a forecast. | cutoff, anchor |
| **horizon** | The number of days after the origin that a forecast covers. | window, lead |
| **backtest** | The rolling-origin evaluation of all forecasters. | validation (for forecasts), simulation |

### 3.2 Technical verbs

| Verb | Meaning |
|---|---|
| **prepare** | Validate the accident file, make the features and split by date. |
| **train** | Fit each pipeline on the train split and select one on the valid split. |
| **score** | Turn one observation into a live score. |
| **convert** | Change a value to the training unit (Fahrenheit, inches of mercury, miles, mph, inches). |
| **map** | Change a weather text or an OpenWeather id to a weather category. |
| **forecast** | Give daily counts for the horizon after an origin. |
| **backtest** | Run the forecasters at each origin and score them on the next days. |
