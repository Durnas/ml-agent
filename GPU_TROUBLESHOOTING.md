# GPU passthrough w Kubernetesie na WSL2 — dziennik diagnostyczny

## Historyczne potwierdzenie problemu

Ok. 2 miesiące przed tą sesją diagnostyczną podejmowana była próba treningu z użyciem KubeRay + Kueue na tym samym komputerze. Według relacji autora: przypisanie GPU w manifeście RayCluster/Kueue nie działało, w efekcie czego zasób zmieniono na CPU, i dopiero ta wersja (CPU) zakończyła się sukcesem (pody osiągały `Running`). To niezależne, wcześniejsze potwierdzenie tego samego ograniczenia opisanego w tym dokumencie — problem nie jest nowy ani jednorazowy, występuje konsekwentnie od co najmniej 2 miesięcy, niezależnie od użytego narzędzia orkiestrującego (KubeRay/Kueue wtedy, bezpośrednie Joby/GPU Operator dziś).

## Cel

`create_training_job` w agencie tworzy Job z `resources.limits: {"nvidia.com/gpu": "1"}`. Żeby taki Job w ogóle mógł zostać zaplanowany, klaster Kubernetes musi zgłaszać `nvidia.com/gpu` jako zasób alokowalny na węźle. Ten dokument śledzi próbę doprowadzenia do tego stanu na Windows 11 + WSL2 (Ubuntu) + karta NVIDIA GeForce RTX 3070.

## Środowisko

- Windows 11, WSL2 (dystrybucja "Ubuntu")
- NVIDIA GeForce RTX 3070, sterownik z obsługą WSL już zainstalowany (`nvidia-smi` na hoście działa)
- Początkowo: Docker Desktop z włączoną integracją WSL2

---

## Próba 1: Minikube + Docker Desktop (WSL2)

**Setup:** `minikube start --driver docker --container-runtime docker --gpus all`, zgodnie z oficjalną dokumentacją Minikube.

**Objaw:** Pody żądające GPU wisiały wiecznie w `Pending`.

### Diagnoza krok po kroku

| Krok | Komenda | Wynik |
|---|---|---|
| 1 | `docker run --rm --gpus all nvidia/cuda:... nvidia-smi` | ✅ Działa — Docker Desktop widzi kartę bezpośrednio |
| 2 | `kubectl describe node minikube \| grep Allocatable` | ❌ Brak `nvidia.com/gpu` |
| 3 | `kubectl get pods -n kube-system \| grep nvidia` | ❌ Brak jakiegokolwiek poda NVIDIA — plugin nigdy nie był zainstalowany |
| 4 | `minikube addons enable nvidia-device-plugin` | Plugin zainstalowany, ale: `CrashLoopBackOff` |
| 5 | Logi pluginu | `Incompatible strategy detected auto` — brak zarejestrowanego runtime "nvidia" w Dockerze węzła |
| 6 | Ręczna rejestracja runtime "nvidia" w `/etc/docker/daemon.json` węzła + `RuntimeClass` + patch `runtimeClassName: nvidia` na DaemonSet | Nowy błąd (postęp!): `failed to initialize NVML: ERROR_LIBRARY_NOT_FOUND` |
| 7 | `minikube ssh -- find / -name "libnvidia-ml.so*"` | **Zero wyników.** Biblioteka NVML nie istnieje NIGDZIE wewnątrz kontenera-węzła Minikube |

### Wniosek

Docker Desktop przekazuje GPU do kontenerów uruchamianych *bezpośrednio* (`docker run`), ale **nie propaguje bibliotek GPU do zagnieżdżonych kontenerów** (kontener-węzeł Minikube, sam będąc kontenerem Docker Desktop, nie dziedziczy tego dostępu dla tego co jest wewnątrz niego). To ograniczenie architektoniczne Docker Desktop na WSL2, nie błąd konfiguracji do poprawienia.

Potwierdzone źródłami: [Lune.dev — Why Isn't Kubernetes Detecting My NVIDIA GPU on Docker Desktop with WSL2](https://www.lune.dev/questions/4771/why-isnt-kubernetes-detecting-my-nvidia-gpu-on-docker-desktop-with-wsl), [NVIDIA/k8s-device-plugin#646](https://github.com/NVIDIA/k8s-device-plugin/issues/646).

---

## Próba 2: Natywny Docker Engine + K3s (bez Docker Desktop)

**Uzasadnienie:** znaleziony, potwierdzony working setup (Andrey Krisanov, [akrisanov.com/wsl-nvidia-gpu](https://akrisanov.com/wsl-nvidia-gpu/)) używa natywnego silnika (nie Docker Desktop) + K3s. Backup pełnej dystrybucji WSL2 zrobiony przed zmianami (`C:\wsl-backups\ubuntu-backup-2026-07-28.tar`) na wypadek problemów.

### Kroki wykonane (wszystkie zweryfikowane, każdy zakończony sukcesem)

1. **Instalacja natywnego Docker Engine** w WSL2 (apt, oficjalne repo Dockera) — ✅ `docker run hello-world` działa
2. **Instalacja `nvidia-container-toolkit`** (tym razem naprawdę się zainstalował, `nvidia-ctk` obecny — poprzednio nie było go wcale, bo Docker Desktop ma własny, wewnętrzny mechanizm) — ✅
3. **`nvidia-ctk runtime configure --runtime=docker` + restart Dockera** — ✅ `docker run --gpus all nvidia-smi` działa **przez prawdziwy toolkit**, nie przez skrót Docker Desktop
4. **Instalacja K3s** (`curl -sfL https://get.k3s.io | sh -`) — ✅, scalony kubeconfig pokazuje oba klastry (Minikube + K3s jako `default`)
5. **`nvidia-ctk runtime configure --runtime=containerd` + restart K3s** — ✅
6. **Instalacja NVIDIA device pluginu (v0.17.1, potem v0.18.2) + `RuntimeClass` + patch `runtimeClassName`** — pod wstaje **`1/1 Running`**, bez `CrashLoopBackOff` (duży postęp względem Minikube!)

### Obecny blokujący problem

Logi pluginu pokazują **udaną** rejestrację:
```
Starting to serve 'nvidia.com/gpu' on /var/lib/kubelet/device-plugins/nvidia-gpu.sock
Registered device plugin for 'nvidia.com/gpu' with Kubelet
```
Logi kubeleta (`journalctl -u k3s`) **potwierdzają odbiór**:
```
"Got registration request from device plugin with resource" resourceName="nvidia.com/gpu"
```
Ale zaraz potem, w kolejnym, długotrwałym połączeniu gRPC (`ListAndWatch` — pluginu na bieżąco raportujący dostępność urządzeń kubeletowi):
```
"ListAndWatch ended unexpectedly for device plugin" err="rpc error: code = Unavailable desc = error reading from server: EOF"
```

Efekt: `kubectl describe node` **nigdy** nie pokazuje `nvidia.com/gpu` w `Capacity`/`Allocatable`, mimo poprawnej rejestracji. Sprawdzone: socket istnieje we właściwym miejscu, plugin nie wchodzi w crash-loop (0 restartów po ~6 minutach), zmiana wersji pluginu (v0.17.1 → v0.18.2) nie pomogła.

### To jest potwierdzony, nierozwiązany bug

Identyczny błąd, identyczny kontekst (K3s + containerd): [NVIDIA/k8s-device-plugin#368](https://github.com/NVIDIA/k8s-device-plugin/issues/368) — zgłoszenie **zamknięte przez maintainerów NVIDII jako "not planned"**, nigdy nie rozwiązane. To nie jest coś, co da się naprawić konfiguracją z naszej strony — problem siedzi w implementacji samego pluginu.

## Próba 3: NVIDIA GPU Operator (Helm)

**Uzasadnienie:** inny, pełniejszy produkt NVIDII niż sam bare device-plugin, inna ścieżka inicjalizacji, więc realna szansa na ominięcie buga z Próby 2.

### Przebieg (każdy problem zdiagnozowany i naprawiony)

1. Sprzątnięcie ręcznej instalacji pluginu + RuntimeClass z Próby 2 (żeby się nie gryzły z automatyzacją Operatora)
2. Instalacja Helm, `helm install gpu-operator ... --set driver.enabled=false` (WSL2 już dostarcza dostęp do GPU, Operator nie powinien instalować własnego sterownika) — Helm zgłasza `STATUS: deployed`
3. **Problem:** żaden z faktycznych komponentów GPU (toolkit, device-plugin, dcgm-exporter) się nie wdrożył — tylko orkiestracja. `ClusterPolicy` status: `"No GPU node found"`. **Przyczyna:** `node-feature-discovery` wykrywa karty przez skanowanie magistrali PCI, a WSL2 udostępnia GPU przez zwirtualizowane `/dev/dxg`, nie przez standardowy PCI — auto-detekcja go nie widzi.
   **Naprawa:** ręczne oznaczenie węzła etykietą, którą normalnie dodałby NFD: `kubectl label node durus feature.node.kubernetes.io/pci-10de.present=true` — to udokumentowany, legalny mechanizm obejścia nieudanej auto-detekcji. Zadziałało — Operator zaczął wdrażać właściwe komponenty.
4. **Problem:** `nvidia-container-toolkit-daemonset` i `nvidia-operator-validator` w `Init:CreateContainerError`. Błąd: `path "/" is mounted on "/" but it is not a shared or slave mount`. **Przyczyna:** domyślna propagacja mountu root filesystem w WSL2 to "private", a kontenery wymagające dynamicznego dostępu do zmian w systemie plików hosta potrzebują "shared"/"slave".
   **Naprawa:** `sudo mount --make-rshared /` na hoście + restart zawieszonych podów. Zadziałało — pody zaczęły się poprawnie inicjalizować.
5. **Wynik:** `gpu-feature-discovery`, `nvidia-dcgm-exporter`, `nvidia-device-plugin-daemonset`, `nvidia-operator-validator`, `nvidia-cuda-validator` — wszystkie `1/1 Running` lub `Completed`. Tylko `nvidia-container-toolkit-daemonset` zostaje w `CrashLoopBackOff` (`no NVIDIA devices found` przy próbie tworzenia klasycznych węzłów `/dev/nvidiaN` — inny, prawdopodobnie nieistotny dla reszty mechanizm niż CDI, którego faktycznie używa device-plugin).

Kluczowa różnica względem Próby 2: **ten device-plugin (dołączony przez Operatora, nowsza wersja) ma wbudowaną, natywną obsługę WSL2** — w logach: `Detected platform: wsl`, `Selecting /host/dev/dxg as /dev/dxg`, poprawne znalezienie `libnvidia-ml.so.1` pod właściwą, WSL-ową ścieżką sterownika (`/usr/lib/wsl/drivers/...`), udana generacja CDI spec, **czysta rejestracja bez błędów**.

### Ostateczne ustalenie

Mimo idealnie czystej, WSL-świadomej rejestracji, `kubectl describe node` **nadal nigdy** nie pokazuje `nvidia.com/gpu` w `Capacity`/`Allocatable`. Logi kubeleta (`journalctl -u k3s`) ujawniają dlaczego:

```
"Got registration request from device plugin with resource" resourceName="nvidia.com/gpu"
... (chwilę później) ...
"ListAndWatch ended unexpectedly for device plugin" err="rpc error: code = Unavailable desc = error reading from server: EOF" resource="nvidia.com/gpu"
```

**To dokładnie ten sam błąd co w Próbie 2** — tym razem odtworzony z zupełnie innym, bardziej zaawansowanym binarnym pluginem. To rozstrzyga: problem nie leży w kodzie konkretnego device-pluginu (sprawdziliśmy dwa różne), tylko w warstwie niżej — w sposobie, w jaki `containerd`/kubelet w tym środowisku (K3s na WSL2) obsługuje długotrwałe połączenie gRPC między pluginem a kubeletem. Rejestracja (krótkie wywołanie RPC) przechodzi za każdym razem; strumień `ListAndWatch` (długotrwały, na którym opiera się właściwa aktualizacja `Capacity`) zrywa się z `EOF` zanim zdąży cokolwiek przekazać.

## Próba 4: MicroK8s

**Uzasadnienie:** trzecia niezależna dystrybucja Kubernetesa, z zupełnie innym kubeletem/agentem węzła niż K3s. Komentarze w [NVIDIA/k8s-device-plugin#368](https://github.com/NVIDIA/k8s-device-plugin/issues/368) sugerowały, że błąd `ListAndWatch...EOF` mógł być specyficzny dla implementacji kubeleta w K3s, nie uniwersalny dla WSL2 — MicroK8s (własny, inny kubelet w postaci `kubelite`) był ostatnią rozsądną szansą na obejście tego konkretnego bugu.

### Przebieg — sześć kolejnych, niezależnych blokerów, każdy zdiagnozowany i naprawiony

1. **Hostname z wielkimi literami.** MicroK8s wymaga poprawnej nazwy DNS-label dla węzła; hostname WSL2 domyślnie dziedziczy nazwę komputera Windows ("Durus") przy każdym starcie, nadpisując `hostnamectl set-hostname`. **Naprawa:** jawny wpis w `/etc/wsl.conf`:
   ```ini
   [network]
   hostname = durus
   generateHosts = false
   ```
   plus pełny restart WSL2 (`wsl --shutdown` z Windows, nie z wnętrza dystrybucji).

2. **Brak systemd.** `snap`/`microk8s status --wait-ready` wisiał w nieskończoność z błędem `timeout waiting for snap system profiles to get updated` — `snapd` wymaga systemd do zarządzania profilami apparmor/cgroup, a WSL2 domyślnie go nie włącza. **Naprawa:** dopisanie do `/etc/wsl.conf`:
   ```ini
   [boot]
   systemd=true
   ```
   plus kolejny pełny restart WSL2. Zweryfikowane przez `ps -p 1 -o comm=` zwracające `systemd`.

3. **Port 10250 zajęty przez zombie-proces K3s.** Nawet po naprawie hostname/systemd węzeł nigdy się nie rejestrował (`kubectl get nodes` → `No resources found`), a `kubelite` w kółko crashował z `listen tcp 0.0.0.0:10250: bind: address already in use`. Cały towarzyszący szum w logach (setki linii `grpc: addrConn.createTransport failed... kine.sock`) okazał się efektem ubocznym tego restart-loopa, nie osobnym problemem z dqlite. **Przyczyna:** proces `k3s-server` z Próby 2/3, wciąż uruchomiony jako usługa systemd, nigdy poprawnie nie zatrzymany, blokował port. **Naprawa:** `sudo systemctl stop k3s && sudo systemctl disable k3s`.

4. **Mount propagation (ten sam błąd co w Próbie 3).** `nvidia-container-toolkit-daemonset` → `Init:CreateContainerError`, `path "/" is mounted on "/" but it is not a shared or slave mount`. **Naprawa:** `sudo mount --make-rshared /` (identyczna jak w Próbie 3, tymczasowa — nie przetrwa restartu WSL2).

5. **CDI + brak `libdxcore.so` w specyfikacji.** Po naprawie mountu, toolkit wciąż padał: `unable to install toolkit: ... no NVIDIA devices found`. MicroK8s GPU Operator domyślnie używa trybu CDI (`spec.cdi.enabled: true` w zasobie `ClusterPolicy`). Znaleziony pasujący, wciąż otwarty (marzec 2026) bug: [NVIDIA/nvidia-container-toolkit#1739](https://github.com/NVIDIA/nvidia-container-toolkit/issues/1739) — `nvidia-ctk cdi generate` nie znajduje `libdxcore.so` na WSL2, mimo że plik fizycznie istnieje pod `/usr/lib/wsl/lib/libdxcore.so`, bo narzędzie nie zna WSL-owych ścieżek. **Obejście:** wyłączenie CDI na rzecz trybu legacy: `kubectl patch clusterpolicy cluster-policy --type merge -p '{"spec":{"cdi":{"enabled":false}}}'`. Po tym `nvidia-container-toolkit-daemonset` przeszedł w `1/1 Running` — pierwszy raz w całej historii tego dziennika, że komponent toolkit realnie wystartował i został.

6. **`nvidia-operator-validator` nie znajduje `nvidia-smi`.** Mimo działającego toolkita, walidator (`toolkit-validation` init-container) wpadał w `CrashLoopBackOff`: `exec: "nvidia-smi": executable file not found in $PATH`. Zdiagnozowane dogłębnie: `nvidia-smi` fizycznie istnieje na hoście (`/usr/lib/wsl/lib/nvidia-smi`), ale (a) `/run/nvidia/driver` — katalog, który `driver-validation` normalnie wypełnia symlinkami do binariów sterownika dla innych komponentów — zostaje **całkowicie pusty** w trybie "pre-installed driver" na WSL2 (walidator sterownika po prostu deklaruje sukces bez kopiowania czegokolwiek); (b) katalog specyfikacji CSV (`/etc/nvidia-container-runtime/host-files-for-container.d/`), który normalnie mówi hookowi runtime co wstrzykiwać do kontenerów żądających GPU, w ogóle nie istniał — bo MicroK8s instaluje własny, odizolowany toolkit wewnątrz DaemonSetu, nigdy nie wywołując natywnego `nvidia-ctk runtime configure` na hoście WSL2, które normalnie generuje ten plik automatycznie. **Próby naprawy:** (i) ręczne skopiowanie `nvidia-smi`+`libnvidia-ml.so.1` do `/run/nvidia/driver/` — bez efektu (kontener walidatora `toolkit-validation` w ogóle nie montuje tej ścieżki); (ii) ręczne utworzenie pliku CSV pod `/etc/nvidia-container-runtime/host-files-for-container.d/wsl.csv` z pełną listą bibliotek WSL2 (`libdxcore.so`, `libnvidia-ml.so.1`, `libcuda.so*`, `/dev/dxg`, `nvidia-smi`) — **również bez efektu**, identyczny błąd natychmiast po restarcie poda. **Wniosek:** krok walidacyjny `nvidia-validator --component=toolkit` sprawdza obecność `nvidia-smi` bezpośrednio na PATH swojego własnego kontenera (obraz `nvcr.io/nvidia/gpu-operator`), bez żadnego mechanizmu, który mógłby mu ten plik dostarczyć w tej konkretnej konfiguracji WSL2 — ani przez wolumen hosta, ani przez hook CSV/CDI (oba przetestowane).

### Wynik

Pierwszy raz w czterech próbach realny komponent toolkitu (`nvidia-container-toolkit-daemonset`) wystartował i pozostał `1/1 Running` — więcej postępu niż w Próbach 1-3 na tym konkretnym etapie. Ale `nvidia-operator-validator` (wymagany przez Operatora, zanim jakikolwiek dalszy komponent — w tym `nvidia-device-plugin-daemonset`, kluczowy dla zgłoszenia `nvidia.com/gpu` kubeletowi — zostanie odblokowany) utyka na piątym z rzędu, coraz głębszym problemie ścieżek plików specyficznych dla WSL2. Nie doszliśmy do etapu `ListAndWatch` z Prób 2-3, więc nie potwierdzono ani nie wykluczono, czy MicroK8s ominąłby ten konkretny bug — utknęliśmy o warstwę wcześniej, na fundamentalnie innym (ale tej samej kategorii) problemie.

## Podsumowanie

Cztery niezależne podejścia (Minikube+Docker Desktop, K3s+ręczny plugin, K3s+GPU Operator, MicroK8s+GPU Operator), każde zdiagnozowane do konkretnej, potwierdzonej przyczyny źródłowej, żadne nie doprowadziło do działającego `nvidia.com/gpu` na węźle. Dwie z tych przyczyn (`ListAndWatch...EOF` w Próbach 2-3, brak `libdxcore.so`/`nvidia-smi` na PATH w Próbie 4) są udokumentowane jako otwarte, nierozwiązane zgłoszenia w oficjalnych repozytoriach NVIDII ([NVIDIA/k8s-device-plugin#368](https://github.com/NVIDIA/k8s-device-plugin/issues/368), [NVIDIA/nvidia-container-toolkit#1739](https://github.com/NVIDIA/nvidia-container-toolkit/issues/1739)). Wzór jest spójny: za każdym razem, gdy jedna warstwa stosu GPU-w-Kubernetesie zostaje naprawiona (rejestracja pluginu, mount propagation, CDI), ujawnia się kolejna, głębsza warstwa tego samego zjawiska — narzędzia NVIDII konsekwentnie zakładają standardowe, natywne linuksowe ścieżki sterownika (PCI, `/usr/lib/x86_64-linux-gnu/`, itd.), których WSL2 (sterownik przez `/dev/dxg`, biblioteki pod `/usr/lib/wsl/lib/`, most `libdxcore.so`) po prostu nie dostarcza w sposób, jaki te narzędzia przewidują. To nie jest błąd konfiguracji możliwy do naprawienia z poziomu użytkownika ani z poziomu kodu tego projektu — to udokumentowane, wielowarstwowe ograniczenie środowiska WSL2 jako platformy dla GPU-w-Kubernetesie, potwierdzone niezależnie w czterech różnych dystrybucjach Kubernetesa i dwóch różnych trybach toolkita (CDI, legacy/CSV).
