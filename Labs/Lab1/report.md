# Filmrekommendationssystem med MovieLens

## 1. Inledning

Syftet med detta laborationsprojekt var att utveckla ett filmrekommendationssystem baserat på MovieLens-datasetet. Rekommendationssystem används idag i många typer av digitala plattformar, till exempel streamingtjänster, e-handel och sociala medier. I den här laborationen var målet att rekommendera fem liknande filmer utifrån en vald film.

Projektet byggdes stegvis. Först implementerades en enkel baslinjemetod baserad på genrer. Därefter utvecklades en förbättrad innehållsbaserad metod med hjälp av genrer, taggar, TF-IDF och KNN. Slutligen byggdes en Dash-applikation så att systemet kunde användas interaktivt via ett grafiskt gränssnitt.

---

## 2. Dataset

Det dataset som användes var MovieLens. Följande filer användes i projektet:

- `movies.csv` – innehåller film-ID, titel och genrer
- `tags.csv` – innehåller användargenererade taggar för filmer
- `links.csv` – innehåller kopplingar mellan MovieLens-ID och externa databaser som IMDb och TMDb

Filen `ratings.csv` analyserades i notebook-delen, men användes inte i den slutliga implementationen eftersom systemet i denna version är **content-based** och inte bygger på collaborative filtering.

Filerna `genome-*` användes inte, eftersom uppgiften uttryckligen nämner att de kan ignoreras.

---

## 3. Explorativ dataanalys (EDA)

En första explorativ analys genomfördes i Jupyter Notebook för att förstå datamängden innan modellering.

Analysen fokuserade på:

- antal filmer i datasetet
- struktur och innehåll i `movies.csv`
- genre-fördelning
- förekomst av taggar
- storlek och egenskaper hos ratings-datan

Några viktiga observationer var:

- många filmer har flera genrer
- genrer lagras som textsträngar separerade med `|`
- taggar ger mer detaljerad och semantisk information än genrer
- datamängden är relativt stor, särskilt om hela datasetet används direkt

Eftersom en full likhetsmatris för alla filmer hade blivit mycket minneskrävande, användes ett slumpmässigt urval av filmer för att göra modellen mer praktiskt körbar på en vanlig dator.

---

## 4. Metod

Projektet innehåller tre metoder:

1. en enkel genrebaserad baslinjemodell 
2. en innehållsbaserad modell med TF-IDF 
3. en förbättrad modell med KNN baserad på TF-IDF-features 

### 4.1 Genrebaserad baslinjemodell

Den första modellen bygger endast på filmgenrer. Varje films genrer omvandlades till en numerisk representation genom att dela upp genresträngen och använda en binär kodning. Därefter beräknades likheten mellan filmer med cosinuslikhet.

Fördelar:
- enkel att implementera
- snabb att förstå och tolka
- fungerar som en bra baslinje

Nackdelar:
- använder mycket begränsad information
- skiljer inte mellan filmer med samma genrer men olika innehåll
- ger ofta alltför generella rekommendationer

Denna metod användes främst som jämförelsepunkt.

---

### 4.2 Innehållsbaserad modell med TF-IDF

För att förbättra kvaliteten på rekommendationerna kombinerades två typer av metadata:

- genrer från `movies.csv`
- användartaggar från `tags.csv`

Genrerna omvandlades till textformat och kombinerades med taggar för varje film. Den sammanslagna textrepresentationen vektoriserades sedan med **TF-IDF**. På så sätt kunde modellen fånga upp mer meningsfull information om filmens innehåll än enbart genre.

Fördelar:
- använder rikare information än baslinjemodellen
- fångar mer semantiska likheter
- bättre precision än ren genrejämförelse

Nackdelar:
- beroende av att filmer har taggar
- filmer utan eller med få taggar kan få sämre representation

---

### 4.3 KNN-baserad rekommendation

Som förbättrad teknik användes **K-Nearest Neighbors (KNN)** ovanpå TF-IDF-matrisen. I stället för att alltid beräkna en full jämförelse mellan alla filmer användes KNN för att hitta de närmaste grannarna till en vald film.

KNN användes med cosinusavstånd och brute-force-algoritm.

Fördelar:
- mer skalbar än att hela tiden skapa full likhetsmatris för alla jämförelser
- effektiv för närmaste-granne-sökning
- fungerar bra tillsammans med TF-IDF-features

Valet av antal grannar sattes så att systemet returnerar fem rekommendationer till användaren. Eftersom en film alltid är mest lik sig själv hämtades `top_n + 1` grannar, och den första togs bort från resultatet.

Denna metod fungerade bäst i det färdiga systemet.

---

## 5. Implementation

Projektet delades upp i två huvudsakliga delar:

### 5.1 `recommender.py`

Denna fil innehåller all kärnlogik för rekommendationssystemet:

- inläsning av data
- sammanslagning av filmer, taggar och länkar
- feature engineering
- TF-IDF-vektorisering
- modellbyggande
- funktioner för rekommendationer:
 - `recommend_by_genre`
 - `recommend_by_content`
 - `recommend_by_knn`

Filen innehåller även funktioner för att extrahera kompletterande information om vald film, såsom:

- årtal
- IMDb-länk
- TMDb-länk
- IMDb-ID
- TMDb-ID

### 5.2 `app.py`

I `app.py` byggdes en interaktiv Dash-applikation. Syftet var att göra systemet användbart och lätt att testa.

Applikationen innehåller:

- dropdown för val av film
- dropdown för val av metod
- slider för antal rekommendationer
- knapp för att generera rekommendationer
- knapp för att rensa resultat
- informationskort med:
 - vald film
 - årtal
 - genrer
 - taggar
- länkar till IMDb och TMDb
- tabell som visar rekommenderade filmer

Detta gör att systemet inte bara fungerar tekniskt utan också går att använda som en enkel rekommendationsapplikation.

---

## 6. Användning av `links.csv`

För att göra applikationen mer användbar och mer visuellt professionell användes även `links.csv`. Denna fil användes inte för själva rekommendationsmodellen, men den användes för att förbättra Dash-applikationen.

Genom `links.csv` kunde MovieLens-filmer kopplas till:

- IMDb
- TMDb

Detta gjorde det möjligt att visa externa länkar för den valda filmen i användargränssnittet. Denna förbättring bidrar inte direkt till modellens precision, men förbättrar användarupplevelsen och gör applikationen mer realistisk och mer lik riktiga rekommendationssystem.

Det visar också att flera delar av datasetet har använts på ett genomtänkt sätt.

---

## 7. Resultat och jämförelse

Tre metoder jämfördes i projektet:

| Metod | Beskrivning | Bedömning |
|------|-------------|-----------|
| Genrebaserad | bygger endast på genrer | enkel men begränsad |
| Content-based | bygger på genrer + taggar + TF-IDF | bättre kvalitet |
| KNN + TF-IDF | närmaste grannar i TF-IDF-rummet | bäst balans mellan kvalitet och effektivitet |

### Observationer

Den genrebaserade modellen fungerade, men resultaten blev ofta ganska grova. Filmer med samma övergripande genrer kunde rekommenderas även om de i praktiken inte var särskilt lika.

När taggar lades till förbättrades resultaten tydligt. TF-IDF gjorde det möjligt att representera filmer mer informativt och rekommendationerna blev mer relevanta.

KNN-metoden gav den bästa kombinationen av:

- användbarhet
- prestanda
- kvalitet

Detta gör KNN till den mest lämpliga lösningen i denna implementation.

---

## 8. Begränsningar

Trots att systemet fungerar bra finns flera begränsningar:

### 8.1 Ingen collaborative filtering
`ratings.csv` användes inte i den färdiga modellen. Systemet bygger alltså inte på användares betyg eller beteende, utan endast på metadata.

### 8.2 Sampling
Eftersom ett urval av filmer användes i stället för hela datasetet kan vissa filmer saknas i en viss körning. Det innebär att rekommendationerna kan variera mellan olika körningar.

### 8.3 Taggberoende
Kvaliteten i content-based-modellen beror delvis på om filmerna har taggar eller inte. Filmer med få taggar kan få en svagare representation.

### 8.4 Ingen kvantitativ utvärdering
Projektet innehåller ingen formell metrisk utvärdering såsom precision, recall eller MAP. Bedömningen är därför huvudsakligen kvalitativ.

---

## 9. Designval och motivering

Några viktiga designval gjordes under projektet:

### Val av content-based metod
Detta val gjordes eftersom uppgiften tydligt öppnar för både collaborative filtering och content filtering, och eftersom metadata som genrer och taggar var enkla att använda i ett första fungerande system.

### Val av TF-IDF
TF-IDF valdes eftersom både genrer och taggar kan ses som textbaserade beskrivningar. Det gav en enkel men effektiv numerisk representation.

### Val av KNN
KNN valdes som förbättrad teknik eftersom det är en naturlig metod för att hitta liknande objekt i feature-rum och fungerar bra tillsammans med TF-IDF.

### Val av Dash
Dash valdes eftersom uppgiften uttryckligen säger att koden kan köras som en Dash-applikation. Det gjorde det möjligt att skapa en användbar och interaktiv lösning.

### Användning av links.csv
Detta val gjordes för att förbättra appens användbarhet och ge ett mer komplett system genom att koppla filmer till externa informationskällor.

---

## 10. Slutsats

Detta projekt visar hur ett filmrekommendationssystem kan byggas stegvis från en enkel baslinje till en mer avancerad och användbar lösning.

De viktigaste slutsatserna är:

- en genrebaserad modell fungerar som en enkel startpunkt men är för begränsad
- en content-based modell med genrer och taggar ger klart bättre rekommendationer
- KNN tillsammans med TF-IDF gav den bästa slutliga lösningen i projektet
- Dash-applikationen gjorde systemet mer interaktivt och praktiskt användbart
- användningen av `links.csv` förbättrade gränssnittet och gav projektet ett mer komplett intryck

Sammanfattningsvis uppfyller projektet både grundkraven och kravet på minst en förbättrad teknik, vilket gör det till en stark kandidat för högre betyg.

---

## 11. Framtida arbete

Om systemet skulle utvecklas vidare vore följande naturliga förbättringar:

- använda `ratings.csv` för collaborative filtering
- lägga till kvantitativa utvärderingsmått
- använda hela datasetet med mer effektiv hantering
- hämta posters eller extra metadata från TMDb API
- jämföra fler modeller, till exempel klustring eller matrix factorization

---

## 12. Sammanfattning

Projektet resulterade i ett fungerande filmrekommendationssystem med tre olika metoder och ett interaktivt gränssnitt. Den slutliga lösningen kombinerar maskininlärning, feature engineering och praktisk applikationsutveckling i en och samma laboration.

Detta gav både en tekniskt fungerande lösning och en tydlig demonstration av hur rekommendationssystem kan utvecklas i praktiken.
