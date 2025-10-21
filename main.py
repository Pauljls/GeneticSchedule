import numpy as np
import pygad
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass

# ========== CONFIGURACIÓN DEL HORARIO ==========
BLOQUES = [
    "07:20-08:05", "08:05-08:50", "08:50-09:35", "09:35-10:20",
    "10:20-10:40", "10:40-11:25", "11:25-12:10", "12:10-12:55",
    "12:55-13:25", "13:25-14:10", "14:10-14:55"
]
HORAS_PEDAGOGICAS = [
    "07:20-08:05", "08:05-08:50", "08:50-09:35", "09:35-10:20",
    "10:40-11:25", "11:25-12:10", "12:10-12:55",
    "13:25-14:10", "14:10-14:55"
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
    
    def __init__(self, docentes: List[Docente], grados: List[Grado], 
                 bloques: List[Bloque], num_generaciones: int = 100):
        self.docentes = {d.id: d for d in docentes}
        self.grados = {g.id: g for g in grados}
        self.bloques = {b.id: b for b in bloques}
        self.grados_secciones = [g.nombre for g in grados]  # Definir primero
        self.num_generaciones = num_generaciones
        
        # Pesos de penalización
        self.w_huecos = 10.0
        self.w_conflictos = 1000.0
        self.w_horas = 500.0
        self.w_disponibilidad = 800.0
        
        # Cromosoma: 45 posiciones por grado
        self.horas_por_grado = 45
        self.longitud_cromosoma = self.horas_por_grado * len(self.grados_secciones)  # Ahora sí existe
        
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
            self.w_huecos * self._calc_huecos(solution) +
            self.w_conflictos * self._calc_conflictos(solution) +
            self.w_horas * self._calc_horas(solution) +
            self.w_disponibilidad * self._calc_disponibilidad(solution)
        )
        
        return -penalizacion_total
    
    def _calc_huecos(self, cromosoma: np.ndarray) -> float:
        """
        Penaliza bloques vacíos entre la primera y última clase de un docente por día
        """
        penalizacion = 0.0
        
        for docente_id in self.docentes.keys():
            # Agrupar asignaciones del docente por día y grado
            asignaciones_por_dia = {}  # {(grado, día): [horas]}
            
            for grado_idx, grado_seccion in enumerate(self.grados_secciones):
                inicio = grado_idx * self.horas_por_grado
                fin = inicio + self.horas_por_grado
                grado_cromosoma = cromosoma[inicio:fin]
                
                for indice_local, bloque_id in enumerate(grado_cromosoma):
                    bloque_id = int(bloque_id)
                    if bloque_id >= 0 and bloque_id in self.bloques:
                        bloque = self.bloques[bloque_id]
                        if bloque.docente_id == docente_id:
                            dia, hora = self.indice_a_dia_hora(indice_local)
                            key = (grado_seccion, dia)
                            if key not in asignaciones_por_dia:
                                asignaciones_por_dia[key] = []
                            asignaciones_por_dia[key].append(hora)
            
            # Calcular huecos
            for (grado, dia), horas in asignaciones_por_dia.items():
                if len(horas) > 1:
                    horas_sorted = sorted(horas)
                    rango = horas_sorted[-1] - horas_sorted[0] + 1
                    huecos = rango - len(horas_sorted)
                    penalizacion += huecos
        
        return penalizacion
    
    def _calc_conflictos(self, cromosoma: np.ndarray) -> float:
        """
        Penaliza cuando dos docentes tienen clases en el mismo aula/horario
        """
        penalizacion = 0.0
        ocupacion = {}  # (aula, día, hora) -> lista de docente_ids
        
        for grado_idx, grado_seccion in enumerate(self.grados_secciones):
            inicio = grado_idx * self.horas_por_grado
            fin = inicio + self.horas_por_grado
            grado_cromosoma = cromosoma[inicio:fin]
            
            grado = self.grados[grado_idx]
            
            for indice_local, bloque_id in enumerate(grado_cromosoma):
                bloque_id = int(bloque_id)
                if bloque_id >= 0 and bloque_id in self.bloques:
                    bloque = self.bloques[bloque_id]
                    
                    dia, hora = self.indice_a_dia_hora(indice_local)
                    key = (grado.aula, dia, hora)
                    
                    if key not in ocupacion:
                        ocupacion[key] = []
                    ocupacion[key].append((grado_seccion, bloque.docente_id))
        
        # Detectar conflictos: múltiples docentes en misma aula/hora
        for key, docentes_lista in ocupacion.items():
            if len(docentes_lista) > 1:
                # Verificar si hay docentes diferentes
                docentes_unicos = set(d_id for _, d_id in docentes_lista)
                if len(docentes_unicos) > 1:
                    penalizacion += (len(docentes_unicos) - 1) * 10
        
        return penalizacion
    
    def _calc_horas(self, cromosoma: np.ndarray) -> float:
        """
        Penaliza diferencias entre horas asignadas y horas requeridas de cada docente
        """
        penalizacion = 0.0
        horas_asignadas = {d_id: 0 for d_id in self.docentes.keys()}
        
        for grado_idx in range(len(self.grados_secciones)):
            inicio = grado_idx * self.horas_por_grado
            fin = inicio + self.horas_por_grado
            grado_cromosoma = cromosoma[inicio:fin]
            
            for bloque_id in grado_cromosoma:
                bloque_id = int(bloque_id)
                if bloque_id >= 0 and bloque_id in self.bloques:
                    bloque = self.bloques[bloque_id]
                    horas_asignadas[bloque.docente_id] += bloque.duracion
        
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
        """Genera población inicial respetando disponibilidad"""
        poblacion = []
        
        for _ in range(num_soluciones):
            cromosoma = -np.ones(self.longitud_cromosoma, dtype=int)
            
            # Intentar asignar bloques respetando disponibilidad
            for grado_idx in range(len(self.grados_secciones)):
                inicio = grado_idx * self.horas_por_grado
                fin = inicio + self.horas_por_grado
                
                # Slots disponibles para este grado
                for bloque in self.bloques.values():
                    docente = self.docentes[bloque.docente_id]
                    
                    # Intentar colocar el bloque en posiciones aleatorias válidas
                    for intento in range(10):
                        indice_local = np.random.randint(0, self.horas_por_grado - bloque.duracion + 1)
                        
                        # Verificar disponibilidad para todas las horas del bloque
                        valido = True
                        for h in range(bloque.duracion):
                            dia, hora = self.indice_a_dia_hora(indice_local + h)
                            if not docente.disponibilidad[dia, hora]:
                                valido = False
                                break
                        
                        if valido:
                            # Asignar el bloque
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
            suppress_warnings=True
        )
        
        print("\n🧬 Iniciando optimización genética...")
        ga_instance.run()
        
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        
        print(f"\n✅ Optimización completada!")
        print(f"Fitness: {solution_fitness:.2f}")
        print(f"Penalización: {-solution_fitness:.2f}")
        
        return solution, ga_instance
    
    def generar_horarios_por_grado(self, solucion: np.ndarray) -> Dict[str, pd.DataFrame]:
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
                columns=["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
            )
            
            horarios[grado_seccion] = df
        
        return horarios
    
    def imprimir_horarios(self, solucion: np.ndarray):
        """Imprime horarios por grado"""
        horarios = self.generar_horarios_por_grado(solucion)
        
        print("\n" + "="*100)
        print("📚 HORARIOS POR GRADO")
        print("="*100)
        
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
    disponibilidad_parcial = np.ones((NUM_DIAS, HORAS_POR_DIA), dtype=bool)
    disponibilidad_parcial[0, 5:] = False  # Lunes tarde
    disponibilidad_parcial[3, 5:] = False  # Jueves tarde
    
    # Docentes
    docentes = [
        Docente(0, "Edwin García", "nombrado", 10, disponibilidad_completa),
        Docente(1, "Carlos Mendoza", "contratado", 8, disponibilidad_parcial),
        Docente(2, "María López", "nombrado", 9, disponibilidad_completa),
    ]
    
    # Grados (contenedores de estudiantes)
    grados = [
        Grado(0, 3, "A", "AULA_3A"),
        Grado(1, 3, "B", "AULA_3B"),
        Grado(2, 4, "A", "AULA_4A"),
    ]
    
    # Bloques de clases (1-3 horas)
    # Cada bloque define: qué docente enseña qué materia en qué grado
    bloques = [
        # Grado 3A
        Bloque(0, docente_id=0, grado_id=0, materia="Matemática", duracion=2),
        Bloque(1, docente_id=0, grado_id=0, materia="Matemática", duracion=2),
        Bloque(2, docente_id=1, grado_id=0, materia="Computación", duracion=3),
        Bloque(3, docente_id=1, grado_id=0, materia="Computación", duracion=2),
        
        # Grado 3B
        Bloque(4, docente_id=2, grado_id=1, materia="Comunicación", duracion=3),
        Bloque(5, docente_id=2, grado_id=1, materia="Comunicación", duracion=2),
        
        # Grado 4A
        Bloque(6, docente_id=0, grado_id=2, materia="Física", duracion=3),
        Bloque(7, docente_id=2, grado_id=2, materia="Historia", duracion=2),
    ]
    
    # Ejecutar
    scheduler = GeneticSchedulerV2(
        docentes=docentes,
        grados=grados,
        bloques=bloques,
        num_generaciones=100
    )
    
    mejor_solucion, ga_instance = scheduler.ejecutar()
    scheduler.imprimir_horarios(mejor_solucion)