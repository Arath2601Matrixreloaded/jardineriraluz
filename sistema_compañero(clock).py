# sistema_companero.py
# Simulador de Sustitución de Páginas - Sistema Compañero (Segunda Oportunidad / Clock)

from typing import List, Any, Dict, Tuple

def simulate_second_chance(reference_string: List[Any], frames: int, verbose: bool = True) -> Dict[str, Any]:
    """
    Simula el algoritmo Sistema Compañero (Segunda Oportunidad / Clock).
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
    frame_vals: List[Any] = [None] * frames            # contenido de marcos
    use_bits: List[int] = [0] * frames                  # bits de uso (segunda oportunidad)
    hand = 0                                            # puntero del reloj
    page_faults = 0
    hits = 0
    replacements = 0
    steps: List[Tuple[Any, bool, List[Any], List[int], int]] = []

    def snapshot() -> Tuple[List[Any], List[int]]:
        return (frame_vals.copy(), use_bits.copy())

    # Encabezado bonito
    if verbose:
        print("\n=== SIMULACIÓN: Sistema Compañero (Segunda Oportunidad / Clock) ===")
        print(f"Marcos: {frames}")
        print(f"Referencia: {reference_string}\n")
        cols = ["Paso", "Ref", "Hit/Miss", "Marcos", "Bits uso", "Puntero (pos. tras la acción)"]
        print("{:<5} {:<5} {:<8} {:<20} {:<15} {:<7}".format(*cols))
        print("-"*80)

    paso = 1
    for ref in reference_string:
        if ref in frame_vals:
            hits += 1
            idx = frame_vals.index(ref)
            use_bits[idx] = 1
            frames_state, bits_state = snapshot()
            if verbose:
                print("{:<5} {:<5} {:<8} {:<20} {:<15} {:<7}".format(
                    paso, str(ref), "HIT", str(frames_state), str(bits_state), hand))
            steps.append((ref, True, frames_state, bits_state, hand))
        else:
            page_faults += 1
            placed = False
            while not placed:
                if frame_vals[hand] is None:
                    frame_vals[hand] = ref
                    use_bits[hand] = 1
                    hand = (hand + 1) % frames
                    placed = True
                else:
                    if use_bits[hand] == 0:
                        frame_vals[hand] = ref
                        use_bits[hand] = 1
                        hand = (hand + 1) % frames
                        replacements += 1
                        placed = True
                    else:
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
        print(f"HITS: {hits}  |  Page Faults: {page_faults}  |  Reemplazos: {replacements}")
        print(f"Hit rate: {hit_rate:.2%}  |  Fault rate: {fault_rate:.2%}")
        print("Nota: 'Reemplazos' cuenta las veces que se expulsó una página ya ocupando un marco (no incluye llenado inicial).")

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
    Convierte un string de referencias en lista.
    Acepta formatos como:
      - '7 0 1 2 0 3 0 4 2 3 0 3 2' (separados por espacio)
      - '7,0,1,2,0,3,0,4,2,3,0,3,2' (comas)
      - 'A B C A D E A B' (letras)
    """
    s = s.strip()
    sep = "," if "," in s else " "
    raw = [t for t in s.replace(",", " ").split() if t]
    out: List[Any] = []
    for token in raw:
        try:
            out.append(int(token))
        except ValueError:
            out.append(token)
    return out

def main():
    print("=== Algoritmo de Sustitución de Páginas: Sistema Compañero (Segunda Oportunidad / Clock) ===")
    print("Ingresa la cadena de referencias (ejemplos):")
    print("  7 0 1 2 0 3 0 4 2 3 0 3 2")
    print("  A B C A D E A B A C D B A")
    ref_str = input("Referencias: ").strip()
    if not ref_str:
        # Ejemplo por defecto (clásico)
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
