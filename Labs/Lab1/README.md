# Lab 1 – Filmrekommendationssystem

## Beskrivning

I denna laboration implementerades ett filmrekommendationssystem baserat på MovieLens-datasetet. Målet var att utforska och jämföra olika metoder för att rekommendera filmer samt att bygga en interaktiv applikation.

---

## Syfte

Syftet med laborationen är att:

- implementera olika rekommendationsmetoder 
- analysera deras prestanda 
- bygga ett praktiskt och användarvänligt system 
- demonstrera förståelse för rekommendationssystem 

---

## Dataset

Följande dataset användes:

- `movies.csv` – innehåller filmer och genrer 
- `tags.csv` – användargenererade taggar 
- `links.csv` – kopplingar till externa databaser (t.ex. IMDb, TMDb) 

 `ratings.csv` ingår inte i repositoryt på grund av filstorlek.

---

## Metoder

Tre olika rekommendationsmetoder implementerades:

### 1. KNN + TF-IDF
- Textrepresentation av innehåll (genres + tags)
- Likhet beräknas med cosine similarity
- Ger de bästa resultaten i projektet

### 2. Content-based filtering
- Likhet mellan filmer baserat på innehåll
- Enkel men effektiv metod

### 3. Genre-baserad baseline
- Jämförelse baserat endast på genrer
- Används som referensmodell

---

## Dataförberedelse

- Samplade datasetet för att minska beräkningskostnad 
- Kombinerade genres och tags till en textrepresentation 
- Hanterade saknade värden 
- Transformerade text med TF-IDF 

---

## Implementation

Projektet består av följande filer:

- `recommender.py` – innehåller modell och rekommendationslogik 
- `app.py` – Dash-applikation för användargränssnitt 
- `recommender.ipynb` – analys och experiment 
- `report.md` – detaljerad rapport 

---

## Dash-applikation

En interaktiv applikation byggdes med Dash där användaren kan:

- välja en film 
- välja metod 
- ange antal rekommendationer 
- se resultat i en tabell 

Dessutom visas:

- filmens genres 
- taggar 
- länkar till externa sidor (via `links.csv`) 

---

## Användning av links.csv

`links.csv` användes för att förbättra användarupplevelsen genom att:

- koppla filmer till externa databaser 
- möjliggöra öppning av IMDb/TMDb-länkar 
- göra systemet mer realistiskt och komplett 

---

## Resultat

- KNN + TF-IDF gav de bästa rekommendationerna 
- Content-based fungerade bra men något sämre 
- Genre-metoden var enkel men begränsad 

---

## Slutsats

Projektet visar att:

- innehållsbaserade metoder fungerar väl utan användardata 
- kombination av tags och genres förbättrar resultat 
- interaktiva system ger bättre användarupplevelse 

---

## Framtida arbete

- använda `ratings.csv` för collaborative filtering 
- implementera hybridmodeller 
- förbättra ranking av rekommendationer 
- optimera prestanda för större dataset 

---

## ⚙️ Installation

Install dependencies using:

```bash
pip install -r requirements.txt

---

## Körning

För att köra applikationen:

```bash
python app.py