# sistema_companero.py
# Simulación del Sistema Compañero (Buddy System)
# Descripción: Este programa simula la asignación y liberación de memoria
# usando el algoritmo Buddy (sistema compañero). Permite ver división (split)
# y fusión (merge) de bloques en potencias de 2.

from typing import Dict, List, Tuple
import math

class BuddySystem:
    """
    Clase que simula el Buddy System:
      - total_memory: tamaño total en KB (debe ser potencia de 2 multiplicada por min_block_size)
      - min_block_size: tamaño mínimo de bloque en KB (potencia base)
    """

    def __init__(self, total_memory: int, min_block_size: int = 32):
        if total_memory <= 0 or min_block_size <= 0:
            raise ValueError("total_memory y min_block_size deben ser mayores que 0.")
        if total_memory % min_block_size != 0:
            raise ValueError("total_memory debe ser múltiplo de min_block_size.")

        self.total_memory = total_memory
        self.min_block_size = min_block_size

        # Calculamos el número máximo de "órdenes"
        # order = 0 -> tamaño = min_block_size
        # order = max_order -> tamaño = min_block_size * 2**max_order = total_memory
        self.max_order = int(math.log2(self.total_memory // self.min_block_size))

        # free_lists: para cada order guardamos una lista de direcciones base libres
        # inicialmente todo el bloque máximo está libre (dirección 0)
        self.free_lists: Dict[int, List[int]] = {i: [] for i in range(self.max_order + 1)}
        self.free_lists[self.max_order] = [0]

        # allocated_blocks: map process -> (addr, order)
        self.allocated_blocks: Dict[str, Tuple[int, int]] = {}

    def _block_size(self, order: int) -> int:
        """Devuelve el tamaño en KB de un bloque para un orden dado."""
        return self.min_block_size * (2 ** order)

    def _order_for_size(self, size: int) -> int:
        """Devuelve el mínimo order tal que block_size >= size."""
        if size <= 0:
            raise ValueError("Tamaño solicitado debe ser mayor que 0.")
        order = 0
        while self._block_size(order) < size:
            order += 1
            if order > self.max_order:
                raise ValueError("Tamaño solicitado excede la memoria total.")
        return order

    def allocate(self, process: str, size: int) -> bool:
        """
        Intenta asignar un bloque lo más ajustado posible a 'size' (KB) para 'process'.
        Retorna True si se asignó, False si no hay memoria suficiente.
        """
        print(f"\nSolicitando {size} KB para proceso '{process}'...")
        try:
            order_needed = self._order_for_size(size)
        except ValueError as e:
            print("x", e)
            return False

        # Buscar orden disponible >= order_needed
        found_order = None
        for o in range(order_needed, self.max_order + 1):
            if self.free_lists[o]:
                found_order = o
                break

        if found_order is None:
            print("x No hay bloques libres suficientes para satisfacer la petición.")
            return False

        # Si el bloque encontrado es mayor, lo dividimos hasta el order_needed
        addr = self.free_lists[found_order].pop(0)
        while found_order > order_needed:
            found_order -= 1
            # al dividir, generamos dos buddies: addr y addr + block_size(found_order)
            buddy_addr = addr + self._block_size(found_order)
            # Añadimos el buddy libre en la lista del nuevo orden
            self.free_lists[found_order].append(buddy_addr)
            # el bloque que seguimos dividiendo será 'addr' en el siguiente ciclo
            print(f"  -> Dividido: creado buddy en {buddy_addr} (tamaño {self._block_size(found_order)} KB)")

        # Asignamos el bloque final a process
        self.allocated_blocks[process] = (addr, order_needed)
        print(f" Asignado: proceso '{process}' -> dirección {addr}, tamaño {self._block_size(order_needed)} KB (order {order_needed})")
        return True

    def free(self, process: str) -> bool:
        """
        Libera el bloque asignado al proceso y realiza fusiones (merge) con buddies libres.
        Retorna True si se liberó, False si el proceso no tenía asignación.
        """
        if process not in self.allocated_blocks:
            print(f"\n El proceso '{process}' no tiene memoria asignada.")
            return False

        addr, order = self.allocated_blocks.pop(process)
        print(f"\nLiberando proceso '{process}' -> dirección {addr}, tamaño {self._block_size(order)} KB (order {order})")

        # Intentamos fusionar iterativamente con su buddy
        current_addr = addr
        current_order = order

        while True:
            buddy_addr = self._buddy_address(current_addr, current_order)
            free_list = self.free_lists[current_order]
            if buddy_addr in free_list:
                # Si el buddy está libre, lo removemos y subimos de orden
                print(f"  -> Buddy libre encontrado en {buddy_addr} (order {current_order}). Fusionando...")
                free_list.remove(buddy_addr)
                current_addr = min(current_addr, buddy_addr)
                current_order += 1
                if current_order > self.max_order:
                    # Ya alcanzamos el bloque máximo
                    break
            else:
                # No hay buddy libre para fusionar, lo insertamos en la lista y terminamos
                self.free_lists[current_order].append(current_addr)
                print(f"  -> No se encontró buddy libre. Bloque añadido a free_list order {current_order} en addr {current_addr}.")
                break

        # Si salimos con current_order > order y no insertamos (porque fusionamos hasta arriba),
        # debemos asegurarnos de que el bloque resultante esté en la lista correspondiente.
        if current_order > self.max_order:
            # Esto sólo ocurre si fusionamos más allá (teórico), no debería pasar si se validó correctamente.
            current_order = self.max_order
            self.free_lists[current_order].append(0)

        print(f" Liberación completada. Bloque disponible: dirección {current_addr}, tamaño {self._block_size(current_order)} KB (order {current_order})")
        return True

    def _buddy_address(self, addr: int, order: int) -> int:
        """
        Calcula la dirección del buddy usando XOR con el tamaño del bloque.
        Esto asume direcciones base alineadas al tamaño del bloque.
        """
        return addr ^ self._block_size(order)

    def show_memory(self):
        """Imprime el estado actual de las listas libres y los bloques asignados."""
        print("\n=== Estado de memoria (Buddy System) ===")
        print(f"Memoria total: {self.total_memory} KB, tamaño mínimo: {self.min_block_size} KB")
        for o in range(self.max_order, -1, -1):
            size = self._block_size(o)
            free = sorted(self.free_lists[o])
            print(f"Order {o} -> Bloques {size} KB : libres = {free}")
        print(f"Bloques asignados: {self.allocated_blocks}")

def main():
    print("=== SIMULACIÓN: SISTEMA COMPAÑERO (BUDDY SYSTEM) ===")
    try:
        total = int(input("Ingrese memoria total (KB) [por defecto 1024]: ") or 1024)
        min_block = int(input("Ingrese tamaño mínimo de bloque (KB) [por defecto 32]: ") or 32)
    except ValueError:
        print("Entrada inválida. Usando valores por defecto 1024 KB y 32 KB.")
        total = 1024
        min_block = 32

    # Validación simple: total debe ser potencia de 2 * min_block
    if (total // min_block) & ( (total // min_block) - 1 ) != 0:
        print("Advertencia: total/min_block debería ser potencia de 2. Ajustando total al siguiente válido.")
        # Ajustar total al siguiente múltiplo de min_block que haga potencia de 2
        units = 1
        while units < (total // min_block):
            units *= 2
        total = units * min_block
        print(f"Nuevo total ajustado: {total} KB")

    system = BuddySystem(total, min_block)

    while True:
        print("\nOpciones:")
        print("  1) Asignar memoria a proceso")
        print("  2) Liberar memoria de proceso")
        print("  3) Mostrar estado de memoria")
        print("  4) Salir")
        opt = input("Seleccione opción: ").strip()

        if opt == "1":
            pname = input("Nombre del proceso: ").strip()
            try:
                psize = int(input("Tamaño solicitado (KB): ").strip())
            except ValueError:
                print("Tamaño inválido.")
                continue
            system.allocate(pname, psize)
        elif opt == "2":
            pname = input("Nombre del proceso a liberar: ").strip()
            system.free(pname)
        elif opt == "3":
            system.show_memory()
        elif opt == "4":
            print("Saliendo...")
            break
        else:
            print("Opción inválida. Intente nuevamente.")

if __name__ == "__main__":
    main()
