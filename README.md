# Projekt z Algorytmów w Inżynierii Danych
### 1. Treść
Celem projektu jest napisanie własnej biblioteki do automatycznego różniczkowania, przystosowanej do wykorzystania w uczeniu sztucznych sieci neuronowych, w tym przypadku (RNN).
### 2. Wymagania
Należało zaimplementować automatyczne różniczkowanie z akumulacją gradientu wstecz (ang. reverse-mode automatic differentiation). Można zastosować zarówno podejście oparte o graf obliczeniowy lub dynamiczną generację kodu liczącego gradient.
Na ocenę z projektu mają wpływ następujące elementy:

1. efektywność implementacji opracowanej biblioteki,
2. spełnienie zadanej dokładności sieci (min. 85%)
3. zakres i dokładność przeprowadzonych badań i testów
4. ogólna ocena artykułu naukowego (spójność, kwestie edycyjne, stylistyczne etc.)
5. subiektywna ocena implementacji

Należy zmierzyć dokładność nauczonej sieci, przeprowadzić pomiary czasu działania i ilości zaalokowanej pamięci. Warunkiem zaliczenia tej części jest uzyskanie dokładności wyuczenia sieci na poziomie conajmniej 85%. Narzucona architektura sieci oraz konfiguracja i parametry uczenia są przedstawione notatnikach w materiałach pomocniczych poniżej. Na tym etapie prędkość działania implementacji nie będzie miała wpływu na ocenę.
### 3. Artykuł
Artykuł (w języku angielskim) ma opisywać własną implementację biblioteki do automatycznego różniczkowania. Powinien zawierać wcześniej przygotowaną analizę literatur ową, opis przeprowadzonych optymalizacji mających przyśpieszyć działanie biblioteki, testy poprawności, porównanie własnej implementacji z implementacjami referencyjnymi, wnioski oraz bibliografię. Poza dołączonymi poniżej implementacjami referencyjnymi we Flux należy przygotować jeszcze jedno rozwiązanie referencyjne (np. w PyTorch/Keras/Tensorflow) oraz zbadać czas jego działania. Artykuł ma zawierać dokładnie 4 strony tekstu i być wykonany w szablonie IEEE Conference Template. W ISOD poza plikiem PDF z artykułem naukowym należy załączyć kompletny, finalny kod biblioteki. Jest to wersja kodu, którą będzie oceniana na obronie projektu.
Termin oddania w ISOD: T10/T15
### 4. Terminy
1. Opracowanie przeglądu literatury dotyczącego: algorytmów automatycznego różniczkowania, efektywnej implementacji w kontekście sztucznych sieci neuronowych CNN/RNN, języka Julia. Przegląd literatury ma być pierwszą sekcją w finalnym artykule naukowym. Powinien zajmować od pół do jednej strony tekstu (w ww. szablonie). Przegląd literatury powinien być napisany w języku angielskim.
   Termin oddania w ISOD: T7/T12.
2. Własna implementacja biblioteki do automatycznego różniczkowania wskazanej sieci neuronowej: 
   Termin oddania w ISOD: T9/T14.
3. Opracowanie artykułu naukowego. 
4. Obrona projektu 
