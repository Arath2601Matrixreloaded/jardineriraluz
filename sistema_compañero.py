# sistema_companero.py
# Simulador de Administración de Memoria - Sistema Compañero (Algoritmo de Sustitución de Páginas - Segunda Oportunidad)
# Autor: Edsau
# Descripción:
# Este programa simula la sustitución de páginas en memoria utilizando el algoritmo de
# Segunda Oportunidad (también conocido como "Clock"), como parte del estudio del Sistema Compañero.
# El sistema gestiona las páginas en los marcos, otorgando a cada una una "segunda oportunidad"
# antes de ser reemplazada, para optimizar el uso de la memoria principal.

from typing import List, Any, Dict, Tuple

def simulate_second_chance(reference_string: List[Any], frames: int, verbose: bool = True) -> Dict[str, Any]:
    """
    Simula el algoritmo de Sustitución de Páginas - Segunda Oportunidad (Sistema Compañero).
    reference_string: lista con las referencias de páginas (números o letras).
    frames: cantidad de marcos de página.
    verbose: si True imprime tabla paso a paso.

    Retorna un diccionario con:
      - steps: lista de tuplas (ref, hit, frames_state, use_bits_state, hand_after)
      - page_faults, hits, hit_rate, fault_rate, replacements
    """
    if frames <= 0:
        raise ValueError("El número de marcos debe ser mayor que 0.")
    if not reference_string:
        raise ValueError("La cadena de referencias no puede estar vacía.")

    # Estado inicial
    frame_vals: List[Any] = [None] * frames      # Contenido de los marcos
    use_bits: List[int] = [0] * frames           # Bits de uso (segunda oportunidad)
    hand = 0                                     # Puntero tipo reloj
    page_faults = 0
    hits = 0
    replacements = 0
    steps: List[Tuple[Any, bool, List[Any], List[int], int]] = []

    def snapshot() -> Tuple[List[Any], List[int]]:
        return (frame_vals.copy(), use_bits.copy())

    # Encabezado de la simulación
    if verbose:
        print("\n=== SIMULACIÓN: SISTEMA COMPAÑERO - ALGORITMO DE SEGUNDA OPORTUNIDAD ===")
        print(f"Marcos disponibles: {frames}")
        print(f"Cadena de referencias: {reference_string}\n")
        cols = ["Paso", "Ref", "Hit/Miss", "Marcos", "Bits de uso", "Puntero"]
        print("{:<5} {:<5} {:<8} {:<20} {:<15} {:<7}".format(*cols))
        print("-"*80)

    paso = 1
    for ref in reference_string:
        if ref in frame_vals:
            # Caso HIT: la página ya está cargada
            hits += 1
            idx = frame_vals.index(ref)
            use_bits[idx] = 1  # Marca que fue usada recientemente
            frames_state, bits_state = snapshot()
            if verbose:
                print("{:<5} {:<5} {:<8} {:<20} {:<15} {:<7}".format(
                    paso, str(ref), "HIT", str(frames_state), str(bits_state), hand))
            steps.append((ref, True, frames_state, bits_state, hand))
        else:
            # Caso MISS: página no está en memoria, debe cargarse
            page_faults += 1
            placed = False
            while not placed:
                if frame_vals[hand] is None:
                    # Hay un marco libre
                    frame_vals[hand] = ref
                    use_bits[hand] = 1
                    hand = (hand + 1) % frames
                    placed = True
                else:
                    if use_bits[hand] == 0:
                        # Reemplaza la página actual
                        frame_vals[hand] = ref
                        use_bits[hand] = 1
                        hand = (hand + 1) % frames
                        replacements += 1
                        placed = True
                    else:
                        # Da una segunda oportunidad (pone el bit en 0)
                        use_bits[hand] = 0
                        hand = (hand + 1) % frames

            frames_state, bits_state = snapshot()
            if verbose:
                print("{:<5} {:<5} {:<8} {:<20} {:<15} {:<7}".format(
                    paso, str(ref), "MISS", str(frames_state), str(bits_state), hand))
            steps.append((ref, False, frames_state, bits_state, hand))
        paso += 1

    total = len(reference_string)
    hit_rate = hits / total
    fault_rate = page_faults / total

    if verbose:
        print("-"*80)
        print(f"Total referencias: {total}")
        print(f"HITS: {hits}  |  Fallos de página: {page_faults}  |  Reemplazos: {replacements}")
        print(f"Tasa de aciertos (Hit rate): {hit_rate:.2%}  |  Tasa de fallos (Fault rate): {fault_rate:.2%}")
        print("Nota: Los reemplazos cuentan solo cuando se expulsa una página ocupando un marco (no incluye el llenado inicial).")

    return {
        "steps": steps,
        "page_faults": page_faults,
        "hits": hits,
        "hit_rate": hit_rate,
        "fault_rate": fault_rate,
        "replacements": replacements
    }

def parse_reference_string(s: str) -> List[Any]:
    """
    Convierte una cadena de texto en una lista de referencias de páginas.
    Acepta formatos como:
      - '7 0 1 2 0 3 0 4 2 3 0 3 2'
      - 'A B C A D E A B'
    """
    s = s.strip()
    raw = [t for t in s.replace(",", " ").split() if t]
    out: List[Any] = []
    for token in raw:
        try:
            out.append(int(token))
        except ValueError:
            out.append(token)
    return out

def main():
    print("=== ADMINISTRACIÓN DE MEMORIA ===")
    print("Simulación: Sistema Compañero con algoritmo de Sustitución de Páginas - Segunda Oportunidad (Clock)")
    print("Ejemplo de referencia:")
    print("  7 0 1 2 0 3 0 4 2 3 0 3 2")
    ref_str = input("Cadena de referencias: ").strip()
    if not ref_str:
        ref_str = "7 0 1 2 0 3 0 4 2 3 0 3 2"
        print(f"(Usando ejemplo por defecto) {ref_str}")
    refs = parse_reference_string(ref_str)
    try:
        frames = int(input("Número de marcos: ").strip())
    except Exception:
        frames = 3
        print("(Usando 3 marcos por defecto)")

    simulate_second_chance(refs, frames, verbose=True)

if __name__ == "__main__":
    main()
