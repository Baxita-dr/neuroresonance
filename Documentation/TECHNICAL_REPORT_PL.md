# Techniczna kontrola estymatora opisowego PLV dla syntetycznych sygnałów oddechu i fNIRS

**Autorzy:** Bastian Fernandez, Aleksandra Zwolińska  
**Afiliacja:** XIV Liceum Ogólnokształcące we Wrocławiu  
**Wersja oprogramowania:** 1.0.0  
**Wersja raportu:** 1.0 · 15 listopada 2026 r.  
**Typ pracy:** kontrola techniczna na danych syntetycznych; bez udziału ludzi i bez walidacji urządzenia.
**Użycie AI:** asystent pomógł w przeglądzie zgodności kodu i redakcji; autorzy odpowiadają za treść.

## Streszczenie

Przedmiotem pracy jest techniczna kontrola prototypu NeuroResonance, który oblicza opisowy Phase-Locking Value (PLV) pomiędzy jawnie wskazanym kanałem oddechowym a kanałem HbO₂. Nie jest to badanie koordynacji mózgu i naczyń ani walidacja fizjologiczna. Sygnały są wyrównywane według timestampów, filtrowane w paśmie 0,07–0,15 Hz, a ich faza jest wyznaczana transformatą Hilberta. Estymator wymaga 100 sekund wspólnego sygnału. Powtarzalna kontrola inżynierska używa 1000 par wygenerowanych przebiegów i dokładnie tej implementacji estymatora co aplikacja. W przypadku kontrolnym dwóch sinusoid o tej samej częstotliwości mediana PLV wyniosła 99,73% (empiryczne kwantyle 5.–95.: 99,07–99,95%). Dla niezależnych oscylatorów o częstotliwościach losowanych z tego samego pasma mediana wyniosła 14,31%, a kwantyle 5.–95. wyniosły 3,78–92,05%. Szeroki zakres drugiego scenariusza pokazuje, że podobna częstotliwość w skończonym oknie może dać wysokie PLV mimo niezależnego generowania sygnałów. Są to wyłącznie wyniki dla jawnie określonych zabawek syntetycznych; nie stanowią testu istotności, estymaty częstości fałszywych alarmów dla fizjologii ani dowodu sprzężenia. Nie zebrano danych od uczestników ani z urządzenia fNIRS.

**Słowa kluczowe:** fNIRS, oddech, PLV, sygnały syntetyczne, walidacja techniczna.

## 1. Cel i granice wnioskowania

PLV opisuje skupienie różnicy faz dwóch sygnałów w wybranym oknie. W tej pracy pytanie dotyczy tego, czy aplikacja oblicza tę statystykę w sposób powtarzalny dla prostych, znanych przebiegów. Nie pytamy, czy u ludzi występuje koordynacja naczyń z mózgiem. Tego wniosku nie da się wyprowadzić z syntetycznych sygnałów.

fNIRS mierzy zmiany optyczne, które mogą zawierać zarówno komponenty związane z korą, jak i powierzchowne oraz układowe zmiany hemodynamiczne. Oscylacje w pobliżu 0,1 Hz mogą być związane m.in. z falami Mayera. Sama zgodność faz oddechu i HbO₂ nie identyfikuje źródła sygnału, przyczynowości ani udziału kory. Ograniczenia te są dobrze znane w metodologii fNIRS (Tachtsidis i Scholkmann, 2016; Yücel i in., 2016).

Oddychanie w pobliżu 0,1 Hz jest omawiane w literaturze biofeedbacku zmienności rytmu serca, ale nie ustanawia to swoistego sprzężenia sygnału fNIRS z oddechem (Lehrer i Gevirtz, 2014; Zaccaro i in., 2018). Zelano i in. (2016) badali naturalny oddech i wewnątrzczaszkowe sygnały EEG; ich wyniku nie przenosimy na pomiary fNIRS ani na narzucone tempo 0,1 Hz.

## 2. Aplikacja i estymator

NeuroResonance jest prototypem edukacyjnym w Pythonie i Streamlit. Obsługuje cztery źródła:

| Tryb | Oddech | HbO₂/HbR | Zakres interpretacji |
|---|---|---|---|
| SIMULATOR | syntetyczny | syntetyczne | wspólna składowa jest wprowadzona konstrukcyjnie |
| WEBCAM | niekalibrowany wskaźnik ruchu ROI | syntetyczne | nie jest to para pomiarów biologicznych |
| LIVE_HARDWARE | jawnie wskazany kanał LSL, jeśli jest dostępny | jawnie wskazane kanały LSL | semantykę i jednostki musi zweryfikować operator |
| XDF_FILE | jawnie wskazany kanał, jeśli jest dostępny | jawnie wskazane kanały XDF | importer nie rozpoznaje automatycznie znaczenia kanałów |

Wbudowana demonstracja XDF jest syntetyczna i zawiera zadaną wspólną składową 0,1 Hz. Nie jest nagraniem fizjologicznym.

Tor estymacji wykorzystuje timestampy, odrzuca analizę przy niewystarczającym wspólnym zakresie lub wykrytej dużej luce, filtruje sygnały dolnoprzepustowo przy częstotliwości źródłowej przed interpolacją do wspólnej siatki 2 Hz, a następnie stosuje filtr Butterwortha 4. rzędu w paśmie 0,07–0,15 Hz. Filtracja pasmowa używa `sosfiltfilt`; do obliczenia PLV pomijane są do 3 sekund z brzegów okna. Faza jest wyznaczana z sygnału analitycznego.

<!-- PIPELINE_DIAGRAM -->

*Rys. 1. Etapy obliczenia opisowego PLV.*

$$
\mathrm{PLV}=\left|\frac{1}{N}\sum_{t=1}^{N}
\exp\left(i\,[\phi_{oddech}(t)-\phi_{HbO_2}(t)]\right)\right|.
$$

PLV i koherencja Welcha są odrębnymi miarami. Aplikacja pokazuje PLV wyłącznie opisowo: nie przypisuje progów ani stanów fizjologicznych, nie wyznacza wartości p i nie przedstawia różnicy faz jako opóźnienia fizjologicznego. Podsumowania są generowane lokalnie; sygnały nie są wysyłane do zewnętrznego modelu językowego. Opcjonalny zapis LSL jest lokalny, domyślnie wyłączony i niezaszyfrowany.

## 3. Powtarzalna kontrola inżynierska

Kontrolę wykonano przez skrypt `Program/tools/run_synthetic_phase_consistency.py`, który wywołuje produkcyjny `ResonanceEstimator`. Polecenie odtworzenia, uruchamiane z katalogu `Program/`:

```text
python tools/run_synthetic_phase_consistency.py --replicates 1000 --seed 20261115
```

Konfiguracja: Python 3.12, NumPy 2.5.4, SciPy 1.18.1; częstotliwość próbkowania 10 Hz; wygenerowany sygnał 110 s; analizowane jest ostatnie 100 s. Generator używa ziarna 20261115. W każdej replice oddech ma częstotliwość 0,1 Hz i dodany szum normalny o odchyleniu standardowym 0,20. HbO₂ jest jednym z dwóch przebiegów:

1. **Kontrola dodatnia:** sinus 0,1 Hz o stałej w danej replice różnicy faz względem oddechu, z niezależnym szumem normalnym o odchyleniu 0,20.
2. **Oscylator niezależny:** faza i częstotliwość HbO₂ są losowane niezależnie; częstotliwość jest jednostajna w zakresie 0,07–0,15 Hz, a dodany szum ma odchylenie 0,20.

| Scenariusz syntetyczny | Liczba replik | Mediana PLV | Empiryczny kwantyl 5. | Empiryczny kwantyl 95. |
|---|---:|---:|---:|---:|
| Ta sama częstotliwość, stała różnica faz | 1000 | 99,73% | 99,07% | 99,95% |
| Niezależnie wylosowane częstotliwości w tym samym paśmie | 1000 | 14,31% | 3,78% | 92,05% |

Kwantyle opisują rozkład wyników tych replik. Nie są przedziałami ufności dla populacji ani przedziałami predykcji dla sygnałów fizjologicznych. Wysoki kwantyl 95. w scenariuszu niezależnym pokazuje, dlaczego sam PLV ani stały próg nie wystarcza do wnioskowania o synchronizacji. Skrypt nie oblicza istotności statystycznej i nie jest biologicznym modelem zerowym. Surowy wynik JSON znajduje się w `Documentation/synthetic_phase_consistency_seed_20261115.json`.

## 4. Wyniki i interpretacja

Kontrola dodatnia sprawdza oczekiwane zachowanie estymatora wobec dwóch przebiegów o tej samej częstotliwości i niezmiennej różnicy faz. Scenariusz niezależny jest celowo trudniejszy: obie częstotliwości losowane są z pasma analizowanego przez filtr. Gdy w skończonym oknie częstotliwości wylosują się blisko siebie, różnica faz zmienia się powoli i PLV może być wysoki mimo niezależnego generowania.

Wyniki potwierdzają tylko to, że kod daje powtarzalną statystykę dla tych dwóch skonstruowanych przypadków. Nie estymują czułości ani swoistości w badaniu, nie kalibrują błędu I rodzaju i nie uprawniają do progu decyzyjnego. Nie wykonywano testów istotności. Nie raportujemy wcześniejszych wyników z 30-sekundowych okien ani p-value z przesunięć cyklicznych: taka procedura nie jest aktywna w aktualnym kodzie i nie jest częścią tej kontroli.

## 5. Ograniczenia i dalsza walidacja

- Nie ma danych od ludzi, pomiaru referencyjnego oddechu ani testu z urządzeniem fNIRS. Wyników nie można uogólniać na uczestników ani urządzenia.
- Sinusoidy z addytywnym szumem nie odwzorowują artefaktów ruchowych, zmienności osobniczej, opóźnień czujników, utraty próbek, nieliniowości ani wspólnych wpływów krążeniowych.
- Pasmo 0,07–0,15 Hz obejmuje zarówno wolne zmiany oddechowe, jak i oscylacje systemowe; fNIRS samo nie rozdziela ich od komponentu korowego. Wpływ fal Mayera i korzyści z krótkich odległości źródło–detektor wymagają pomiarów dodatkowych, nie samego PLV (Yücel i in., 2016).
- Minimalne okno 100 s jest ograniczeniem konfiguracji i rozdzielczości analizy, a nie dowiedzioną granicą poprawnego wnioskowania.
- Mapowanie kanałów LSL/XDF zależy od metadanych urządzenia i decyzji operatora. Import XDF wczytuje plik do pamięci przed częścią limitów zawartości; limit rozmiaru uploadu nie gwarantuje stałego maksymalnego zużycia RAM.
- Zapis sesji LSL jest zwykłym CSV/JSON. Przed użyciem danych osobowych potrzebne są odpowiednie zgody, zasady dostępu, retencji i usuwania oraz wymagane zatwierdzenia etyczne.

Przed badaniem pilotażowym należy użyć skalibrowanego czujnika oddechu, potwierdzić mapowanie kanałów na rzeczywistym zestawie LSL/XDF, zaplanować kontrolę ruchu i układowych zmian hemodynamicznych oraz z góry opisać hipotezy, wielkość próby, model zerowy i korektę wielokrotnych analiz. Przydatne będą kanały krótkiego rozstawu fNIRS oraz niezależny pomiar układu krążenia, np. PPG/HRV. Synchronizacja sygnałów pozostaje opisowa i nie dowodzi kierunku ani przyczynowości.

## 6. Wniosek

NeuroResonance jest prototypem do wizualizacji sygnałów i obliczania opisowego PLV. Powtarzalna kontrola syntetyczna sprawdza działanie kodu dla jawnie zdefiniowanych przebiegów. Nie stanowi wyniku biologicznego, walidacji klinicznej ani dowodu koordynacji mózgu i naczyń. Program nie służy do diagnozowania ani leczenia.

## Bibliografia

1. Lachaux, J.-P., Rodriguez, E., Martinerie, J., & Varela, F. J. (1999). Measuring phase synchrony in brain signals. *Human Brain Mapping, 8*(4), 194–208. https://doi.org/10.1002/(SICI)1097-0193(1999)8:4<194::AID-HBM4>3.0.CO;2-C
2. Tachtsidis, I., & Scholkmann, F. (2016). False positives and false negatives in functional near-infrared spectroscopy: issues, challenges, and the way forward. *Neurophotonics, 3*(3), 031405. https://doi.org/10.1117/1.NPh.3.3.031405
3. Yücel, M. A., Selb, J., Aasted, C. M., Lin, P.-Y., Borsook, D., Becerra, L., & Boas, D. A. (2016). Mayer waves reduce the accuracy of estimated hemodynamic response functions in functional near-infrared spectroscopy. *Biomedical Optics Express, 7*(8), 3078–3088. https://doi.org/10.1364/BOE.7.003078
4. Lehrer, P. M., & Gevirtz, R. (2014). Heart rate variability biofeedback: how and why does it work? *Frontiers in Psychology, 5*, 756. https://doi.org/10.3389/fpsyg.2014.00756
5. Zaccaro, A., Piarulli, A., Laurino, M., Garbella, E., Menicucci, D., Neri, B., & Gemignani, A. (2018). How breath-control can change your life: a systematic review on psycho-physiological correlates of slow breathing. *Frontiers in Human Neuroscience, 12*, 353. https://doi.org/10.3389/fnhum.2018.00353
6. Zelano, C., Jiang, H., Zhou, G., Arora, N., Schuele, S., Rosenow, J., & Gottfried, J. A. (2016). Nasal respiration entrains human limbic oscillations and modulates cognitive function. *The Journal of Neuroscience, 36*(49), 12448–12467. https://doi.org/10.1523/JNEUROSCI.2586-16.2016

