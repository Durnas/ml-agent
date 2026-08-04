# Kontekst projektu — do wczytania na nowej maszynie

Jeśli zaczynasz nową sesję Claude Code i nie masz historii tej rozmowy, przeczytaj to najpierw.

## Co to za projekt

Praca licencjacka: konwersacyjny agent AI do zarządzania Kubernetesem (nie tylko trening ML — pełne zarządzanie klastrem: pody, deploymenty, service'y, configmapy, RBAC, węzły itd., 47 narzędzi). LLM: Groq (`openai/gpt-oss-120b`), UI: Streamlit.

Dwa równoległe foldery projektu (kod w obu identyczny poza dokumentem poniżej):
- **`ml-agent-mvp-cluster`** — stabilna, główna wersja do oddania. Działa w pełni poprawnie, poza jednym ograniczeniem środowiskowym (patrz niżej).
- **`ml-agent-mvp-gpu`** (ten folder) — kopia robocza do eksperymentów z GPU-w-Kubernetesie, zawiera dodatkowo `GPU_TROUBLESHOOTING.md`.

## Stan: GPU passthrough w Kubernetesie

`create_training_job` (jedno z 47 narzędzi agenta) tworzy Job z `resources.limits: {"nvidia.com/gpu": "1"}`. Żeby to zadziałało, klaster musi zgłaszać `nvidia.com/gpu` jako zasób. **To nie działa na WSL2** (Windows) — przeczytaj `GPU_TROUBLESHOOTING.md` w tym folderze, tam jest pełny dziennik 4 niezależnych prób (Minikube, K3s×2, MicroK8s), każda zdiagnozowana do konkretnego, potwierdzonego, otwartego bugu upstreamowego w narzędziach NVIDII specyficznego dla WSL2 (nie dla samego Kubernetesa/Dockera).

**Dlatego przechodzimy na natywnego Linuksa** — wszystkie napotkane problemy były specyficzne dla WSL2 (nietypowy most `/dev/dxg`, brak standardowej magistrali PCI, `libdxcore.so`), więc na prawdziwym Linuksie nie powinny wystąpić.

## Nowa maszyna (na której prawdopodobnie czytasz ten plik)

Lenovo Y520-15IKBN, i7-7700HQ, 8GB RAM, GTX 1050 (4GB), 1TB HDD, dual-boot Windows+Ubuntu (Ubuntu dostał ~150GB).

### Zrobione już na tej maszynie:
- ✅ Ubuntu zainstalowane (dual-boot obok Windows)
- ✅ Git zainstalowany, repo `ml-agent-mvp-gpu` sklonowane z `https://github.com/Durnas/ml-agent.git`
- ✅ Docker Engine zainstalowany i działa (`docker run hello-world` przeszło)
- ✅ Sterownik NVIDIA działa natywnie: `nvidia-smi` pokazuje GTX 1050, Driver 580.173.02, CUDA 13.0 — **bez żadnych WSL2-owych problemów**

### Następne kroki (w kolejności):
1. Zainstalować `nvidia-container-toolkit`, skonfigurować Docker (`nvidia-ctk runtime configure --runtime=docker`)
2. Zweryfikować: `docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi` — **to kluczowy test**, na WSL2 różne warianty tego zawsze się w końcu wywalały, na natywnym Linuksie powinno przejść od razu
3. Zainstalować K3s (`curl -sfL https://get.k3s.io | sh -`) LUB MicroK8s — polecam K3s jako prostszy, mniej ruchomych części niż MicroK8s
4. Zainstalować NVIDIA device plugin (albo pełny GPU Operator przez Helm) w klastrze
5. Sprawdzić `kubectl describe node | grep nvidia.com/gpu` w sekcji Capacity/Allocatable — to jest ostateczny test sukcesu
6. Jeśli działa: skonfigurować venv (`python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`), odpalić `streamlit run app.py`, przetestować `create_training_job` end-to-end z prawdziwym repo GitHub zawierającym `train.py`
7. Zaktualizować `GPU_TROUBLESHOOTING.md` o wynik (sukces albo nowy, konkretny blocker)

## Ważne zasady pracy (feedback od użytkownika w tej sesji)

- Dokumentować **wszystko** w `GPU_TROUBLESHOOTING.md` — to materiał do pracy licencjackiej, historia porażek jest tak samo wartościowa jak sukces
- Nie hallucynować możliwości agenta (np. nigdy nie sugerować że agent ma dostęp do shella/SSH — nie ma)
- Destrukcyjne akcje agenta zawsze wymagają potwierdzenia w UI — to nie zmienia się nigdy
- Klucz Groq API nigdy nie jest zapisywany na dysku (tylko `st.session_state`, wpisywany ręcznie w UI za każdym razem)
