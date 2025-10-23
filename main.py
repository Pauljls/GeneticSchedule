import numpy as np
import pygad
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass

# ========== CONFIGURACIÓN DEL HORARIO ==========
BLOQUES = [
    "07:20-08:05",
    "08:05-08:50",
    "08:50-09:35",
    "09:35-10:20",
    "10:20-10:40",
    "10:40-11:25",
    "11:25-12:10",
    "12:10-12:55",
    "12:55-13:25",
    "13:25-14:10",
    "14:10-14:55",
]
HORAS_PEDAGOGICAS = [
    "07:20-08:05",
    "08:05-08:50",
    "08:50-09:35",
    "09:35-10:20",
    "10:40-11:25",
    "11:25-12:10",
    "12:10-12:55",
    "13:25-14:10",
    "14:10-14:55",
]
RECREO_INICIO = 4  # Índice 4 = 10:20-10:40
ALMUERZO_INICIO = 8  # Índice 8 = 12:55-13:25
HORAS_POR_DIA = 9  # Horas pedagógicas por día
NUM_DIAS = 5


@dataclass
class Docente:
    id: int
    nombre: str
    tipo: str
    horas_requeridas: int
    disponibilidad: np.ndarray  # Shape: (5 días, 9 bloques)


@dataclass
class Grado:
    """Representa un grado-sección (3A, 3B, etc.)"""

    id: int
    numero: int  # 3, 4, 5, etc.
    seccion: str  # A, B, C, etc.
    aula: str

    @property
    def nombre(self):
        return f"{self.numero}{self.seccion}"


@dataclass
class Bloque:
    """Representa una clase: 1-3 horas consecutivas de una materia con un docente para un grado"""

    id: int
    docente_id: int
    grado_id: int
    materia: str
    duracion: int  # 1, 2 o 3 horas pedagógicas


class GeneticSchedulerV2:
    """
    Nueva estructura:
    - Cromosoma por grado: array de 45 posiciones (0-44)
    - Cada posición = 1 hora pedagógica en el horario semanal
    - Valor en posición = ID del bloque de clase (o -1 si vacío)
    - Cromosoma total = 45 * número_de_grados (135 para 3 grados)

    Índice a (día, hora_pedagogica):
        día = índice // 9
        hora = índice % 9
    """

    def __init__(
        self,
        docentes: List[Docente],
        grados: List[Grado],
        bloques: List[Bloque],
        num_generaciones: int = 100,
    ):
        self.docentes = {d.id: d for d in docentes}
        self.grados = {g.id: g for g in grados}
        self.bloques = {b.id: b for b in bloques}
        self.grados_secciones = [g.nombre for g in grados]  # Definir primero
        self.num_generaciones = num_generaciones

        # Pesos de penalización
        self.w_huecos = 800.0
        self.w_conflictos = 1000.0
        self.w_horas = 500.0
        self.w_disponibilidad = 10.0
        self.w_consecutividad = (
            1000.0  # Restricción dura: bloques deben ser consecutivos
        )

        # Cromosoma: 45 posiciones por grado
        self.horas_por_grado = 45
        self.longitud_cromosoma = self.horas_por_grado * len(
            self.grados_secciones
        )  # Ahora sí existe

        print(f"🧬 Cromosoma inicializado:")
        print(f"   - Horas por grado: {self.horas_por_grado}")
        print(f"   - Grados: {len(self.grados_secciones)}")
        print(f"   - Longitud total: {self.longitud_cromosoma}")
        print(f"   - Bloques disponibles: {len(bloques)}")

    def indice_a_dia_hora(self, indice: int) -> Tuple[int, int]:
        """Convierte índice (0-44) a (día, hora_pedagógica)"""
        dia = indice // HORAS_POR_DIA
        hora = indice % HORAS_POR_DIA
        return dia, hora

    def dia_hora_a_indice(self, dia: int, hora: int) -> int:
        """Convierte (día, hora_pedagógica) a índice (0-44)"""
        return dia * HORAS_POR_DIA + hora

    def obtener_grado_seccion_de_indice(self, indice_global: int) -> Tuple[str, int]:
        """Obtiene grado-sección e índice local del cromosoma global"""
        grado_idx = indice_global // self.horas_por_grado
        indice_local = indice_global % self.horas_por_grado
        return self.grados_secciones[grado_idx], indice_local

    def fitness_function(self, ga_instance, solution, solution_idx):
        """Función de fitness para PyGAD (a MAXIMIZAR)"""
        penalizacion_total = (
            self.w_huecos * self._calc_huecos(solution)
            + self.w_conflictos * self._calc_conflictos(solution)
            + self.w_horas * self._calc_horas(solution)
            + self.w_disponibilidad * self._calc_disponibilidad(solution)
            + 100.0 * self._validar_bloques_consecutivos(solution)  # Nueva penalización
        )

        return -penalizacion_total

    def _calc_huecos(self, cromosoma: np.ndarray) -> float:
        """
        Penaliza bloques vacíos entre la primera y última clase de un docente por día
        CONSIDERANDO TODOS LOS GRADOS donde enseña ese día
        """
        penalizacion = 0.0

        for docente_id in self.docentes.keys():
            # Agrupar asignaciones del docente por día (SIN separar por grado)
            asignaciones_por_dia = {}  # {día: set(horas)}

            for grado_idx in range(len(self.grados_secciones)):
                inicio = grado_idx * self.horas_por_grado
                fin = inicio + self.horas_por_grado
                grado_cromosoma = cromosoma[inicio:fin]

                for indice_local, bloque_id in enumerate(grado_cromosoma):
                    bloque_id = int(bloque_id)
                    if bloque_id >= 0 and bloque_id in self.bloques:
                        bloque = self.bloques[bloque_id]
                        if bloque.docente_id == docente_id:
                            dia, hora = self.indice_a_dia_hora(indice_local)
                            if dia not in asignaciones_por_dia:
                                asignaciones_por_dia[dia] = set()
                            asignaciones_por_dia[dia].add(hora)

            # Calcular huecos por día (considerando TODOS los grados)
            for dia, horas in asignaciones_por_dia.items():
                if len(horas) > 1:
                    horas_sorted = sorted(horas)
                    rango = horas_sorted[-1] - horas_sorted[0] + 1
                    huecos = rango - len(horas_sorted)
                    penalizacion += huecos

        return penalizacion

    def _validar_bloques_consecutivos(self, cromosoma: np.ndarray) -> float:
        """
        Penaliza si un bloque con duración > 1 NO está en posiciones consecutivas
        Ejemplo: Bloque de 2 hrs debe estar en índices [i, i+1], no [i, i+5]
        """
        penalizacion = 0.0

        for grado_idx in range(len(self.grados_secciones)):
            inicio = grado_idx * self.horas_por_grado
            fin = inicio + self.horas_por_grado
            grado_cromosoma = cromosoma[inicio:fin]

            # Agrupar apariciones de cada bloque
            bloques_apariciones = {}  # bloque_id -> [índices]
            for indice_local, bloque_id in enumerate(grado_cromosoma):
                bloque_id = int(bloque_id)
                if bloque_id >= 0 and bloque_id in self.bloques:
                    if bloque_id not in bloques_apariciones:
                        bloques_apariciones[bloque_id] = []
                    bloques_apariciones[bloque_id].append(indice_local)

            # Validar consecutividad
            for bloque_id, indices in bloques_apariciones.items():
                bloque = self.bloques[bloque_id]

                # Verificar cantidad correcta
                if len(indices) != bloque.duracion:
                    # Penalizar si aparece más o menos veces de lo esperado
                    penalizacion += abs(len(indices) - bloque.duracion) * 5
                    continue

                # Verificar que sean consecutivos
                indices_sorted = sorted(indices)
                for i in range(len(indices_sorted) - 1):
                    # Verificar que sean consecutivos en el mismo día
                    dia1, hora1 = self.indice_a_dia_hora(indices_sorted[i])
                    dia2, hora2 = self.indice_a_dia_hora(indices_sorted[i + 1])

                    if dia1 != dia2 or hora2 != hora1 + 1:
                        # No son consecutivos
                        penalizacion += 5

        return penalizacion

    def _calc_conflictos(self, cromosoma: np.ndarray) -> float:
        """
        Penaliza dos tipos de conflictos:
        1. Múltiples docentes en el mismo grado al mismo tiempo
        2. Un docente enseñando en múltiples grados al mismo tiempo
        """
        penalizacion = 0.0

        # CONFLICTO TIPO 1: Múltiples docentes en mismo grado/hora
        # (Imposible físicamente: dos profes no pueden estar en la misma aula)
        for grado_idx, grado_seccion in enumerate(self.grados_secciones):
            inicio = grado_idx * self.horas_por_grado
            fin = inicio + self.horas_por_grado
            grado_cromosoma = cromosoma[inicio:fin]

            ocupacion_grado = {}  # (día, hora) -> [docente_ids]

            for indice_local, bloque_id in enumerate(grado_cromosoma):
                bloque_id = int(bloque_id)
                if bloque_id >= 0 and bloque_id in self.bloques:
                    bloque = self.bloques[bloque_id]
                    dia, hora = self.indice_a_dia_hora(indice_local)
                    key = (dia, hora)

                    if key not in ocupacion_grado:
                        ocupacion_grado[key] = []
                    ocupacion_grado[key].append(bloque.docente_id)

            # Contar conflictos en este grado
            for key, docentes_lista in ocupacion_grado.items():
                docentes_unicos = set(docentes_lista)
                if len(docentes_unicos) > 1:
                    # Más de 1 docente al mismo tiempo en el mismo grado
                    penalizacion += (len(docentes_unicos) - 1) * 10

        # CONFLICTO TIPO 2: Un docente en múltiples grados al mismo tiempo
        ocupacion_docentes = {}  # docente_id -> {(día, hora): [grados]}

        for grado_idx, grado_seccion in enumerate(self.grados_secciones):
            inicio = grado_idx * self.horas_por_grado
            fin = inicio + self.horas_por_grado
            grado_cromosoma = cromosoma[inicio:fin]

            for indice_local, bloque_id in enumerate(grado_cromosoma):
                bloque_id = int(bloque_id)
                if bloque_id >= 0 and bloque_id in self.bloques:
                    bloque = self.bloques[bloque_id]
                    dia, hora = self.indice_a_dia_hora(indice_local)

                    if bloque.docente_id not in ocupacion_docentes:
                        ocupacion_docentes[bloque.docente_id] = {}

                    key = (dia, hora)
                    if key not in ocupacion_docentes[bloque.docente_id]:
                        ocupacion_docentes[bloque.docente_id][key] = []

                    ocupacion_docentes[bloque.docente_id][key].append(grado_seccion)

        # Contar conflictos por docente
        for docente_id, horarios in ocupacion_docentes.items():
            for key, grados_lista in horarios.items():
                grados_unicos = set(grados_lista)
                if len(grados_unicos) > 1:
                    # Docente en múltiples grados al mismo tiempo
                    penalizacion += (len(grados_unicos) - 1) * 10

        return penalizacion

    def _calc_horas(self, cromosoma: np.ndarray) -> float:
        """
        Penaliza diferencias entre horas asignadas y horas requeridas de cada docente
        IMPORTANTE: Cada aparición del bloque cuenta según su duración
        """
        penalizacion = 0.0
        horas_asignadas = {d_id: 0 for d_id in self.docentes.keys()}
        bloques_contados = (
            {}
        )  # (grado_idx, bloque_id) -> bool para evitar contar duplicados

        for grado_idx in range(len(self.grados_secciones)):
            inicio = grado_idx * self.horas_por_grado
            fin = inicio + self.horas_por_grado
            grado_cromosoma = cromosoma[inicio:fin]

            # Contar apariciones únicas de cada bloque en este grado
            bloques_en_grado = set()
            for bloque_id in grado_cromosoma:
                bloque_id = int(bloque_id)
                if bloque_id >= 0 and bloque_id in self.bloques:
                    bloques_en_grado.add(bloque_id)

            # Sumar duración de cada bloque único
            for bloque_id in bloques_en_grado:
                bloque = self.bloques[bloque_id]
                horas_asignadas[bloque.docente_id] += bloque.duracion

        # Calcular diferencias
        for docente_id, docente in self.docentes.items():
            diferencia = abs(horas_asignadas[docente_id] - docente.horas_requeridas)
            penalizacion += diferencia

        return penalizacion

    def _calc_disponibilidad(self, cromosoma: np.ndarray) -> float:
        """
        Penaliza asignaciones en días/horas donde docentes no están disponibles
        """
        penalizacion = 0.0

        for grado_idx in range(len(self.grados_secciones)):
            inicio = grado_idx * self.horas_por_grado
            fin = inicio + self.horas_por_grado
            grado_cromosoma = cromosoma[inicio:fin]

            for indice_local, bloque_id in enumerate(grado_cromosoma):
                bloque_id = int(bloque_id)
                if bloque_id >= 0 and bloque_id in self.bloques:
                    bloque = self.bloques[bloque_id]
                    docente = self.docentes[bloque.docente_id]

                    dia, hora = self.indice_a_dia_hora(indice_local)
                    if not docente.disponibilidad[dia, hora]:
                        penalizacion += 1

        return penalizacion

    def crear_poblacion_inicial(self, num_soluciones: int) -> np.ndarray:
        """Genera población inicial respetando disponibilidad y consecutividad"""
        poblacion = []

        for _ in range(num_soluciones):
            cromosoma = -np.ones(self.longitud_cromosoma, dtype=int)

            # Intentar asignar bloques respetando disponibilidad y duración
            for grado_idx in range(len(self.grados_secciones)):
                inicio = grado_idx * self.horas_por_grado

                # Filtrar bloques para este grado
                bloques_grado = [
                    b for b in self.bloques.values() if b.grado_id == grado_idx
                ]

                for bloque in bloques_grado:
                    docente = self.docentes[bloque.docente_id]

                    # Intentar colocar el bloque en posiciones válidas
                    for intento in range(20):
                        # Elegir posición aleatoria que permita la duración completa
                        max_inicio = self.horas_por_grado - bloque.duracion
                        if max_inicio < 0:
                            break

                        indice_local = np.random.randint(0, max_inicio + 1)

                        # Verificar que todas las posiciones estén libres y disponibles
                        valido = True
                        for h in range(bloque.duracion):
                            pos = indice_local + h
                            dia, hora = self.indice_a_dia_hora(pos)

                            # Verificar disponibilidad del docente
                            if not docente.disponibilidad[dia, hora]:
                                valido = False
                                break

                            # Verificar que la posición esté libre
                            if cromosoma[inicio + pos] != -1:
                                valido = False
                                break

                            # Verificar que no cruce recreo o almuerzo
                            # (bloques deben estar en el mismo día)
                            dia_actual, _ = self.indice_a_dia_hora(pos)
                            if h > 0:
                                dia_anterior, _ = self.indice_a_dia_hora(pos - 1)
                                if dia_actual != dia_anterior:
                                    valido = False
                                    break

                        if valido:
                            # Asignar el bloque en todas sus horas
                            for h in range(bloque.duracion):
                                cromosoma[inicio + indice_local + h] = bloque.id
                            break

            poblacion.append(cromosoma)

        return np.array(poblacion)

    def ejecutar(self):
        """Ejecuta el algoritmo genético"""
        gene_space = list(range(-1, len(self.bloques)))

        ga_instance = pygad.GA(
            num_generations=self.num_generaciones,
            num_parents_mating=10,
            fitness_func=self.fitness_function,
            sol_per_pop=50,
            num_genes=self.longitud_cromosoma,
            gene_space=gene_space,
            gene_type=int,
            parent_selection_type="tournament",
            K_tournament=3,
            crossover_type="single_point",
            crossover_probability=0.8,
            mutation_type="random",
            mutation_probability=0.15,
            keep_elitism=5,
            initial_population=self.crear_poblacion_inicial(50),
            suppress_warnings=True,
        )

        print("\n🧬 Iniciando optimización genética...")
        ga_instance.run()

        solution, solution_fitness, solution_idx = ga_instance.best_solution()

        print(f"\n✅ Optimización completada!")
        print(f"Fitness: {solution_fitness:.2f}")
        print(f"Penalización: {-solution_fitness:.2f}")

        return solution, ga_instance

    def generar_horarios_por_grado(
        self, solucion: np.ndarray
    ) -> Dict[str, pd.DataFrame]:
        """Genera DataFrames de horarios por grado"""
        horarios = {}

        for grado_idx, grado_seccion in enumerate(self.grados_secciones):
            inicio = grado_idx * self.horas_por_grado
            fin = inicio + self.horas_por_grado
            grado_cromosoma = solucion[inicio:fin]

            # Crear matriz horas × días (transponemos para pandas)
            horario_matrix = np.empty((HORAS_POR_DIA, NUM_DIAS), dtype=object)
            horario_matrix[:] = ""

            for indice_local, bloque_id in enumerate(grado_cromosoma):
                bloque_id = int(bloque_id)
                if bloque_id >= 0 and bloque_id in self.bloques:
                    bloque = self.bloques[bloque_id]
                    docente = self.docentes[bloque.docente_id]
                    dia, hora = self.indice_a_dia_hora(indice_local)

                    info = f"{bloque.materia}\n{docente.nombre}"
                    if horario_matrix[hora, dia]:
                        horario_matrix[hora, dia] += f"\n⚠️{info}"
                    else:
                        horario_matrix[hora, dia] = info

            # Convertir a DataFrame
            df = pd.DataFrame(
                horario_matrix,
                index=HORAS_PEDAGOGICAS,
                columns=["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"],
            )

            horarios[grado_seccion] = df

        return horarios

    def imprimir_horarios(self, solucion: np.ndarray):
        """Imprime horarios por grado"""
        horarios = self.generar_horarios_por_grado(solucion)

        print("\n" + "=" * 100)
        print("📚 HORARIOS POR GRADO")
        print("=" * 100)

        for grado_seccion, df in horarios.items():
            print(f"\n{'='*100}")
            print(f"📖 GRADO: {grado_seccion}")
            print(f"{'='*100}")
            print(df.to_string())
            print()


# ========== EJEMPLO DE USO ==========
if __name__ == "__main__":
    # Disponibilidad
    disponibilidad_completa = np.ones((NUM_DIAS, HORAS_POR_DIA), dtype=bool)

    disponibilidad_parcial_1 = np.ones((NUM_DIAS, HORAS_POR_DIA), dtype=bool)
    disponibilidad_parcial_1[0, 5:] = False  # Lunes tarde
    disponibilidad_parcial_1[3, 5:] = False  # Jueves tarde

    disponibilidad_parcial_2 = np.ones((NUM_DIAS, HORAS_POR_DIA), dtype=bool)
    disponibilidad_parcial_2[1, :4] = False  # Martes mañana
    disponibilidad_parcial_2[4, 6:] = False  # Viernes tarde

    # Docentes (más docentes para más materias)
    docentes = [
        Docente(0, "Edwin García", "nombrado", 18, disponibilidad_completa),
        Docente(1, "Carlos Mendoza", "contratado", 15, disponibilidad_parcial_1),
        Docente(2, "María López", "nombrado", 18, disponibilidad_completa),
        Docente(3, "Ana Torres", "nombrado", 15, disponibilidad_completa),
        Docente(4, "Luis Ramírez", "contratado", 12, disponibilidad_parcial_2),
        Docente(5, "Patricia Vega", "nombrado", 18, disponibilidad_completa),
        Docente(6, "Roberto Silva", "contratado", 9, disponibilidad_completa),
    ]

    # Grados (contenedores de estudiantes)
    grados = [
        Grado(0, 3, "A", "AULA_3A"),
        Grado(1, 3, "B", "AULA_3B"),
        Grado(2, 4, "A", "AULA_4A"),
    ]

    # Bloques de clases (1-3 horas) con currículo completo
    # Distribución típica semanal: ~30-35 horas pedagógicas por grado
    bloques = [
        # ========== GRADO 3A (30 horas) ==========
        # Matemática: 6 hrs (Edwin)
        Bloque(0, docente_id=0, grado_id=0, materia="Matemática", duracion=2),
        Bloque(1, docente_id=0, grado_id=0, materia="Matemática", duracion=2),
        Bloque(2, docente_id=0, grado_id=0, materia="Matemática", duracion=2),
        # Comunicación: 6 hrs (María)
        Bloque(3, docente_id=2, grado_id=0, materia="Comunicación", duracion=3),
        Bloque(4, docente_id=2, grado_id=0, materia="Comunicación", duracion=3),
        # Ciencia y Tecnología: 5 hrs (Ana)
        Bloque(5, docente_id=3, grado_id=0, materia="Ciencia y Tecnología", duracion=3),
        Bloque(6, docente_id=3, grado_id=0, materia="Ciencia y Tecnología", duracion=2),
        # Personal Social: 4 hrs (Patricia)
        Bloque(7, docente_id=5, grado_id=0, materia="Personal Social", duracion=2),
        Bloque(8, docente_id=5, grado_id=0, materia="Personal Social", duracion=2),
        # Inglés: 3 hrs (Luis)
        Bloque(9, docente_id=4, grado_id=0, materia="Inglés", duracion=3),
        # Arte: 2 hrs (Roberto)
        Bloque(10, docente_id=6, grado_id=0, materia="Arte", duracion=2),
        # Educación Física: 2 hrs (Roberto)
        Bloque(11, docente_id=6, grado_id=0, materia="Educación Física", duracion=2),
        # Computación: 2 hrs (Carlos)
        Bloque(12, docente_id=1, grado_id=0, materia="Computación", duracion=2),
        # ========== GRADO 3B (30 horas) ==========
        # Matemática: 6 hrs (Edwin)
        Bloque(13, docente_id=0, grado_id=1, materia="Matemática", duracion=2),
        Bloque(14, docente_id=0, grado_id=1, materia="Matemática", duracion=2),
        Bloque(15, docente_id=0, grado_id=1, materia="Matemática", duracion=2),
        # Comunicación: 6 hrs (María)
        Bloque(16, docente_id=2, grado_id=1, materia="Comunicación", duracion=3),
        Bloque(17, docente_id=2, grado_id=1, materia="Comunicación", duracion=3),
        # Ciencia y Tecnología: 5 hrs (Ana)
        Bloque(
            18, docente_id=3, grado_id=1, materia="Ciencia y Tecnología", duracion=3
        ),
        Bloque(
            19, docente_id=3, grado_id=1, materia="Ciencia y Tecnología", duracion=2
        ),
        # Personal Social: 4 hrs (Patricia)
        Bloque(20, docente_id=5, grado_id=1, materia="Personal Social", duracion=2),
        Bloque(21, docente_id=5, grado_id=1, materia="Personal Social", duracion=2),
        # Inglés: 3 hrs (Luis)
        Bloque(22, docente_id=4, grado_id=1, materia="Inglés", duracion=3),
        # Arte: 2 hrs (Roberto)
        Bloque(23, docente_id=6, grado_id=1, materia="Arte", duracion=2),
        # Educación Física: 2 hrs (Carlos)
        Bloque(24, docente_id=1, grado_id=1, materia="Educación Física", duracion=2),
        # Computación: 2 hrs (Carlos)
        Bloque(25, docente_id=1, grado_id=1, materia="Computación", duracion=2),
        # ========== GRADO 4A (32 horas) ==========
        # Matemática: 6 hrs (Edwin)
        Bloque(26, docente_id=0, grado_id=2, materia="Matemática", duracion=2),
        Bloque(27, docente_id=0, grado_id=2, materia="Matemática", duracion=2),
        Bloque(28, docente_id=0, grado_id=2, materia="Matemática", duracion=2),
        # Comunicación: 6 hrs (María)
        Bloque(29, docente_id=2, grado_id=2, materia="Comunicación", duracion=3),
        Bloque(30, docente_id=2, grado_id=2, materia="Comunicación", duracion=3),
        # Ciencia y Tecnología: 6 hrs (Ana)
        Bloque(
            31, docente_id=3, grado_id=2, materia="Ciencia y Tecnología", duracion=3
        ),
        Bloque(
            32, docente_id=3, grado_id=2, materia="Ciencia y Tecnología", duracion=3
        ),
        # Personal Social: 4 hrs (Patricia)
        Bloque(33, docente_id=5, grado_id=2, materia="Personal Social", duracion=2),
        Bloque(34, docente_id=5, grado_id=2, materia="Personal Social", duracion=2),
        # Inglés: 3 hrs (Luis)
        Bloque(35, docente_id=4, grado_id=2, materia="Inglés", duracion=3),
        # Arte: 2 hrs (Roberto)
        Bloque(36, docente_id=6, grado_id=2, materia="Arte", duracion=2),
        # Educación Física: 3 hrs (Carlos)
        Bloque(37, docente_id=1, grado_id=2, materia="Educación Física", duracion=3),
        # Computación: 2 hrs (Carlos)
        Bloque(38, docente_id=1, grado_id=2, materia="Computación", duracion=2),
    ]

    print("\n📊 RESUMEN DE CARGA HORARIA:")
    print("=" * 60)

    # Calcular horas por docente
    horas_por_docente = {d.id: 0 for d in docentes}
    for bloque in bloques:
        horas_por_docente[bloque.docente_id] += bloque.duracion

    for docente in docentes:
        horas_asignadas = horas_por_docente[docente.id]
        diferencia = horas_asignadas - docente.horas_requeridas
        estado = "✓" if diferencia == 0 else f"⚠️ ({diferencia:+d})"
        print(
            f"{docente.nombre:20} - Req: {docente.horas_requeridas:2d}hrs | Asig: {horas_asignadas:2d}hrs {estado}"
        )

    # Calcular horas por grado
    print("\n📚 HORAS POR GRADO:")
    print("=" * 60)
    horas_por_grado = {g.id: 0 for g in grados}
    for bloque in bloques:
        horas_por_grado[bloque.grado_id] += bloque.duracion

    for grado in grados:
        horas = horas_por_grado[grado.id]
        print(f"Grado {grado.nombre:3} - {horas} horas pedagógicas semanales")

    # Ejecutar
    print("\n" + "=" * 60)
    scheduler = GeneticSchedulerV2(
        docentes=docentes, grados=grados, bloques=bloques, num_generaciones=150
    )

    mejor_solucion, ga_instance = scheduler.ejecutar()
    scheduler.imprimir_horarios(mejor_solucion)
