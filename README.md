## Symulacja rozrostu glejaka wielopostaciowego _in vivo_ w oparciu o dane z eksperymentów na myszach

**Glejak wielopostaciowy (_Glioblastoma multiforme_)** jest nowotworem o szczególnie dynamicznym i agresywnym przebiegu. Charakteryzuje się on szybkim tempem wzrostu, wysoką inwazyjnością oraz zdolnością do infiltracji otaczających tkanek mózgu, co czyni go wyzwaniem dla współczesnej onkologii i neurochirurgii. Zrozumienie złożonej dynamiki rozwoju tego nowotworu wymaga analizy nie tylko samych komórek nowotworowych, ale także ich interakcji z mikrośrodowiskiem, w tym z układem naczyniowym i macierzą zewnątrzkomórkową.
<br />

Niniejszy projekt stawia sobie za cel opracowanie i analizę wieloskalowego modelu symulacyjnego, który odwzorowuje wzrost tego guza w warunkach _in vivo_. Integruje on złożone procesy biologiczne w trzech sprzężonych ze sobą skalach.

### Instrukcja uruchomienia

Poniżej przedstawiono sekwencję komend potrzebnych do uruchomienia symulacji na systemie Windows. Wymagane jest posiadanie zainstalowanego interpretera języka Python (wersja >= 3.11.X):

```py
git clone https://github.com/worthy11/tumor-simulation
cd tumor-simulation

py -m venv venv
venv/Scripts/activate
pip install -r requirements.txt
py run.py
```
