# NeuroResonance

Prototyp edukacyjno-badawczy do wizualizacji jawnie wskazanych sygnałów oddechu i fNIRS oraz obliczania opisowego Phase-Locking Value (PLV).

**Zakres dowodów:** kontrola techniczna na sygnałach syntetycznych. W tym projekcie nie ma wyników od uczestników ani walidacji z urządzeniem fNIRS. Program nie ustala źródła korowego, przyczynowości, stanu zdrowia ani diagnozy.

## Co zawiera

- Dashboard Streamlit z trybami SIMULATOR, WEBCAM (niekalibrowany wskaźnik ruchu), LIVE_HARDWARE (LSL) i XDF_FILE.
- Deterministyczny, jawnie syntetyczny przykład demonstracyjny.
- Mapowanie kanałów LSL/XDF wykonywane przez operatora; aplikacja nie rozpoznaje automatycznie ich znaczenia fizjologicznego.
- Estymator opisowego PLV i odrębna miara koherencji Welcha.
- Panel pochodzenia oraz jakości sygnału, lokalne podsumowanie i eksport wybranego okna danych.
- Testy jednostkowe oraz skrypt do odtwarzalnej kontroli dwóch syntetycznych scenariuszy.

## Uruchomienie w Windows

Wymagany jest Python 3.12. Uruchom **Program/run_project.bat**; skrypt tworzy lokalne środowisko i instaluje wersje bibliotek przypięte w **Program/requirements.txt**. Serwer Streamlit nasłuchuje na **127.0.0.1**.

## Odtworzenie wyników syntetycznych

Z katalogu **Program/** uruchom:

```text
python tools/run_synthetic_phase_consistency.py --replicates 1000 --seed 20261115
```

Wyniki raportu zapisano w **Documentation/synthetic_phase_consistency_seed_20261115.json**. Skrypt wykorzystuje produkcyjny estymator. Przebiegi są sztuczne; kontrola nie jest testem istotności ani biologicznym modelem zerowym.

## Testy oprogramowania

Z katalogu **Program/**, po zainstalowaniu zależności projektu i pytest:

```text
python -m pytest tests/ -v
```

Raport nie deklaruje wyniku testów dla bieżącej paczki. Status automatycznego uruchomienia po publikacji należy sprawdzić w zakładce **Actions**.

## Ograniczenia i prywatność

- PLV jest statystyką opisową; aplikacja nie stosuje progów klasyfikacji, wartości p ani opóźnienia fazowego.
- Analiza wymaga co najmniej 100 sekund poprawnego wspólnego sygnału. To wymaganie estymatora, nie próg udowodnionej poprawności fizjologicznej.
- Podsumowania są lokalne; aplikacja nie wysyła sygnałów do zewnętrznego modelu.
- Opcjonalny zapis LSL jest domyślnie wyłączony i niezaszyfrowany. Eksport może zawierać surowe sygnały; przed udostępnieniem sprawdź podstawę prawną, zgodę i anonimizację.
- XDF jest parsowany w pamięci; limit rozmiaru uploadu nie gwarantuje stałego maksymalnego zużycia RAM.

## Struktura

- **Program/src/** — aplikacja i estymator.
- **Program/tests/** — testy automatyczne.
- **Program/tools/run_synthetic_phase_consistency.py** — kontrola syntetyczna.
- **Documentation/TECHNICAL_REPORT_PL.md** — aktualny raport metodologiczny.
- **Documentation/NeuroResonance_Raport_PL.pdf** — wersja PDF raportu.

## Autorzy i cytowanie

Bastian Fernandez, Aleksandra Zwolińska — XIV Liceum Ogólnokształcące we Wrocławiu. Dane o wersji i autorach znajdują się także w **CITATION.cff**.

Repozytorium projektu: https://github.com/Baxita-dr/neuroresonance

## Licencja

MIT. Zobacz plik **LICENSE**.
