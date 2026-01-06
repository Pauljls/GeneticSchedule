"""
Algoritmo Genético para Optimización de Horarios Escolares usando PyGAD
========================================================================
Este módulo implementa un algoritmo genético para generar horarios óptimos
que minimicen las fracciones de horarios vacíos y cumplan con múltiples restricciones.

Utiliza:
- PyGAD: Para el algoritmo genético
- NumPy: Para estructuras de datos eficientes
- Pandas: Para visualización y manejo de datos tabulares
"""

import numpy as np
import pandas as pd
import pygad
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Set
import random
import warnings
warnings.filterwarnings('ignore')

# ========================= CONFIGURACIÓN GLOBAL =========================

HORAS_PEDAGOGICAS = [
    "07:20-08:05", "08:05-08:50", "08:50-09:35", "09:35-10:20",
    "10:40-11:25", "11:25-12:10", "12:10-12:55",
    "13:25-14:10", "14:10-14:55"
]

NUM_AULAS = 3
AULAS = ["3A", "3B", "4"]
POSICIONES_POR_AULA = 45  # 9 horas × 5 días
LONGITUD_CROMOSOMA = 135  # 45 × 3 aulas

# Días de la semana
DIAS = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes"]
NUM_DIAS = 5
HORAS_POR_DIA = 9

# Parámetros del algoritmo genético (optimizados para minimizar horas vacías)
POBLACION_INICIAL = 150  # Aumentado para mayor diversidad
NUM_GENERACIONES = 800   # Más generaciones para mejor convergencia
NUM_PADRES = 30          # Más padres para mayor exploración
TASA_MUTACION = 0.15     # Mayor mutación para escapar de óptimos locales
TASA_CRUCE = 0.85        # Mayor cruce para combinar buenas soluciones

# Pesos para la función de fitness
PESO_CRUCES = 100      # Penalización por cruce de profesores
PESO_DISPONIBILIDAD = 50  # Penalización por violar disponibilidad
PESO_HORAS_VACIAS = 30    # Penalización por horas vacías entre clases
PESO_HORAS_SEMANALES = 80 # Penalización por no cumplir horas semanales

# ========================= ESTRUCTURAS DE DATOS =========================

@dataclass
class Profesor:
    """Representa un profesor en el sistema"""
    id: int
    nombre: str
    codigo: str  # Código único del profesor
    disponibilidad: np.ndarray  # (5 días, 9 horas) - 1 si está disponible, 0 si no
    horas_semanales: int  # Horas que debe dictar
    cursos: List[int] = field(default_factory=list)  # IDs de cursos que enseña
    
    def __post_init__(self):
        """Validar y ajustar la disponibilidad"""
        if self.disponibilidad.shape != (NUM_DIAS, HORAS_POR_DIA):
            self.disponibilidad = np.ones((NUM_DIAS, HORAS_POR_DIA))

@dataclass
class Curso:
    """Representa un curso/materia"""
    id: int
    nombre: str
    codigo: str
    profesor_id: int
    horas_semanales: int
    aula_preferida: Optional[str] = None  # Aula preferida para este curso
    
@dataclass
class Bloque:
    """
    Representa un bloque de horas consecutivas
    Es la unidad mínima indivisible en el horario
    """
    id: int
    curso_id: int
    profesor_id: int
    duracion: int  # Horas consecutivas (1, 2 o 3)
    aula_asignada: Optional[str] = None
    dia: Optional[int] = None
    hora_inicio: Optional[int] = None

# ========================= CLASE PRINCIPAL DEL SISTEMA =========================

class SistemaHorarios:
    """Sistema principal para gestión de horarios con algoritmo genético"""
    
    def __init__(self):
        self.profesores: Dict[int, Profesor] = {}
        self.cursos: Dict[int, Curso] = {}
        self.bloques: List[Bloque] = []
        self.cromosoma_actual: Optional[np.ndarray] = None
        self.mejor_cromosoma: Optional[np.ndarray] = None
        self.mejor_fitness: float = -np.inf
        self.historial_fitness: List[float] = []
        
    def agregar_profesor(self, profesor: Profesor):
        """Agregar un profesor al sistema"""
        self.profesores[profesor.id] = profesor
        
    def agregar_curso(self, curso: Curso):
        """Agregar un curso al sistema"""
        self.cursos[curso.id] = curso
        if curso.profesor_id in self.profesores:
            self.profesores[curso.profesor_id].cursos.append(curso.id)
    
    def generar_bloques(self):
        """Generar bloques de enseñanza basados en cursos y horas semanales"""
        self.bloques = []
        bloque_id = 0
        
        for curso in self.cursos.values():
            horas_restantes = curso.horas_semanales
            
            while horas_restantes > 0:
                # Determinar duración del bloque (preferir bloques de 2 horas)
                if horas_restantes >= 2:
                    duracion = 2
                else:
                    duracion = 1
                    
                bloque = Bloque(
                    id=bloque_id,
                    curso_id=curso.id,
                    profesor_id=curso.profesor_id,
                    duracion=duracion
                )
                
                self.bloques.append(bloque)
                bloque_id += 1
                horas_restantes -= duracion
    
    def indices_globales_a_locales(self, indice_global: int) -> Tuple[int, int, int]:
        """
        Convierte un índice global del cromosoma a índices locales
        
        Args:
            indice_global: Índice en el cromosoma (0-134)
            
        Returns:
            Tupla (aula, dia, hora) donde:
            - aula: 0-2 (índice del aula)
            - dia: 0-4 (índice del día)
            - hora: 0-8 (índice de la hora)
        """
        aula = indice_global // POSICIONES_POR_AULA
        posicion_en_aula = indice_global % POSICIONES_POR_AULA
        dia = posicion_en_aula // HORAS_POR_DIA
        hora = posicion_en_aula % HORAS_POR_DIA
        
        return aula, dia, hora
    
    def indices_locales_a_global(self, aula: int, dia: int, hora: int) -> int:
        """
        Convierte índices locales a un índice global del cromosoma
        
        Args:
            aula: 0-2 (índice del aula)
            dia: 0-4 (índice del día)
            hora: 0-8 (índice de la hora)
            
        Returns:
            Índice global en el cromosoma (0-134)
        """
        return aula * POSICIONES_POR_AULA + dia * HORAS_POR_DIA + hora
    
    def crear_cromosoma_inicial(self) -> np.ndarray:
        """
        Crea un cromosoma inicial válido
        Cada gen representa el ID del bloque asignado a esa posición (-1 si está vacío)
        """
        cromosoma = np.full(LONGITUD_CROMOSOMA, -1, dtype=np.int32)
        bloques_disponibles = list(range(len(self.bloques)))
        
        # Intentar asignar bloques de manera aleatoria pero válida
        for bloque_id in bloques_disponibles:
            bloque = self.bloques[bloque_id]
            profesor = self.profesores[bloque.profesor_id]
            
            # Buscar posiciones válidas para este bloque
            posiciones_validas = []
            
            for aula in range(NUM_AULAS):
                for dia in range(NUM_DIAS):
                    for hora in range(HORAS_POR_DIA - bloque.duracion + 1):
                        # Verificar disponibilidad del profesor
                        disponible = True
                        for h in range(bloque.duracion):
                            if profesor.disponibilidad[dia, hora + h] == 0:
                                disponible = False
                                break
                        
                        if disponible:
                            # Verificar que las posiciones estén libres
                            posiciones_libres = True
                            for h in range(bloque.duracion):
                                idx = self.indices_locales_a_global(aula, dia, hora + h)
                                if cromosoma[idx] != -1:
                                    posiciones_libres = False
                                    break
                            
                            if posiciones_libres:
                                posiciones_validas.append((aula, dia, hora))
            
            # Asignar el bloque a una posición válida aleatoria
            if posiciones_validas:
                aula, dia, hora = random.choice(posiciones_validas)
                for h in range(bloque.duracion):
                    idx = self.indices_locales_a_global(aula, dia, hora + h)
                    cromosoma[idx] = bloque_id
        
        return cromosoma
    
    def calcular_fitness(self, ga_instance, solution, solution_idx) -> float:
        """
        Función de fitness MEJORADA con enfoque híbrido:
        - Restricciones DURAS (cruces y disponibilidad): deben cumplirse obligatoriamente
        - Restricciones BLANDAS (horas vacías): penalización EXPONENCIAL para minimizarlas agresivamente

        Esta función está optimizada específicamente para ELIMINAR BLOQUES VACÍOS
        entre clases de un mismo profesor.
        """
        cromosoma = solution.astype(np.int32)

        # === MÉTRICAS ===
        num_cruces = 0
        num_violaciones_disponibilidad = 0
        total_horas_vacias = 0
        diferencia_horas_semanales = 0

        # 1. Contar cruces de profesores (RESTRICCIÓN DURA)
        for dia in range(NUM_DIAS):
            for hora in range(HORAS_POR_DIA):
                profesores_hora = {}

                for aula in range(NUM_AULAS):
                    idx = self.indices_locales_a_global(aula, dia, hora)
                    if cromosoma[idx] >= 0 and cromosoma[idx] < len(self.bloques):
                        bloque = self.bloques[cromosoma[idx]]
                        prof_id = bloque.profesor_id

                        if prof_id in profesores_hora:
                            num_cruces += 1
                        else:
                            profesores_hora[prof_id] = aula

        # 2. Contar violaciones de disponibilidad (RESTRICCIÓN DURA)
        for idx in range(LONGITUD_CROMOSOMA):
            if cromosoma[idx] >= 0 and cromosoma[idx] < len(self.bloques):
                aula, dia, hora = self.indices_globales_a_locales(idx)
                bloque = self.bloques[cromosoma[idx]]
                profesor = self.profesores[bloque.profesor_id]

                if profesor.disponibilidad[dia, hora] == 0:
                    num_violaciones_disponibilidad += 1

        # 3. Contar horas vacías entre clases (OBJETIVO PRINCIPAL)
        for profesor_id, profesor in self.profesores.items():
            for dia in range(NUM_DIAS):
                horas_profesor = []

                # Recopilar todas las horas donde enseña el profesor
                for hora in range(HORAS_POR_DIA):
                    for aula in range(NUM_AULAS):
                        idx = self.indices_locales_a_global(aula, dia, hora)
                        if cromosoma[idx] >= 0 and cromosoma[idx] < len(self.bloques):
                            bloque = self.bloques[cromosoma[idx]]
                            if bloque.profesor_id == profesor_id:
                                horas_profesor.append(hora)
                                break

                # Calcular horas vacías entre la primera y última clase
                if len(horas_profesor) > 1:
                    horas_profesor.sort()
                    for i in range(len(horas_profesor) - 1):
                        diferencia = horas_profesor[i + 1] - horas_profesor[i] - 1
                        total_horas_vacias += diferencia

        # 4. Calcular diferencia en horas semanales (RESTRICCIÓN BLANDA)
        horas_asignadas = {prof_id: 0 for prof_id in self.profesores.keys()}
        bloques_contados = set()

        for idx in range(LONGITUD_CROMOSOMA):
            if cromosoma[idx] >= 0 and cromosoma[idx] < len(self.bloques):
                if cromosoma[idx] not in bloques_contados:
                    bloques_contados.add(cromosoma[idx])
                    bloque = self.bloques[cromosoma[idx]]
                    horas_asignadas[bloque.profesor_id] += 1

        for profesor_id, profesor in self.profesores.items():
            diferencia_horas_semanales += abs(profesor.horas_semanales - horas_asignadas[profesor_id])

        # === CÁLCULO DE FITNESS ===

        # FASE 1: Verificar restricciones DURAS
        restricciones_duras = num_cruces + num_violaciones_disponibilidad

        if restricciones_duras > 0:
            # Si hay violaciones duras, penalizar pero permitir evolución
            # Aún consideramos horas vacías para diferenciar soluciones con mismas violaciones duras
            penalizacion_duras = restricciones_duras * 500
            penalizacion_vacias_suave = total_horas_vacias * 10  # Penalización suave
            penalizacion_horas_suave = diferencia_horas_semanales * 5

            fitness = 10000 - penalizacion_duras - penalizacion_vacias_suave - penalizacion_horas_suave
            return max(1, fitness)  # Mínimo 1 para evitar 0

        # FASE 2: Sin violaciones duras, optimizar restricciones BLANDAS agresivamente
        # Penalización EXPONENCIAL para horas vacías (objetivo principal: ELIMINARLAS)
        if total_horas_vacias > 0:
            # Usamos base 1.8 para ser muy agresivo contra horas vacías
            penalizacion_vacias = PESO_HORAS_VACIAS * (1.8 ** total_horas_vacias - 1)
        else:
            penalizacion_vacias = 0  # ¡Perfecto! Sin horas vacías

        # Penalización CUADRÁTICA para horas semanales (menos crítica)
        if diferencia_horas_semanales > 0:
            penalizacion_horas = PESO_HORAS_SEMANALES * (diferencia_horas_semanales ** 1.5)
        else:
            penalizacion_horas = 0

        # Calcular fitness final
        penalizacion_total = penalizacion_vacias + penalizacion_horas

        # Normalizar para evitar valores negativos extremos
        fitness = 10000 / (1 + penalizacion_total / 1000)

        return max(0, fitness)
    
    def cromosoma_a_horario(self, cromosoma: np.ndarray) -> Dict[str, pd.DataFrame]:
        """
        Convierte un cromosoma en DataFrames de horarios para visualización
        
        Returns:
            Diccionario con DataFrames para cada aula
        """
        horarios = {}
        
        for aula_idx, aula_nombre in enumerate(AULAS):
            # Crear matriz para el horario
            horario_matriz = []
            
            for hora in range(HORAS_POR_DIA):
                fila = []
                for dia in range(NUM_DIAS):
                    idx = self.indices_locales_a_global(aula_idx, dia, hora)
                    
                    if cromosoma[idx] >= 0 and cromosoma[idx] < len(self.bloques):
                        bloque = self.bloques[cromosoma[idx]]
                        curso = self.cursos[bloque.curso_id]
                        profesor = self.profesores[bloque.profesor_id]
                        fila.append(f"{curso.nombre}\n({profesor.nombre})")
                    else:
                        fila.append("---")
                
                horario_matriz.append(fila)
            
            # Crear DataFrame
            df = pd.DataFrame(
                horario_matriz,
                index=HORAS_PEDAGOGICAS,
                columns=DIAS
            )
            horarios[aula_nombre] = df
        
        return horarios
    
    def horario_profesor(self, cromosoma: np.ndarray, profesor_id: int) -> pd.DataFrame:
        """
        Genera el horario individual de un profesor
        
        Args:
            cromosoma: Solución del algoritmo genético
            profesor_id: ID del profesor
            
        Returns:
            DataFrame con el horario del profesor
        """
        horario_matriz = []
        
        for hora in range(HORAS_POR_DIA):
            fila = []
            for dia in range(NUM_DIAS):
                clase_encontrada = False
                
                for aula_idx in range(NUM_AULAS):
                    idx = self.indices_locales_a_global(aula_idx, dia, hora)
                    
                    if cromosoma[idx] >= 0 and cromosoma[idx] < len(self.bloques):
                        bloque = self.bloques[cromosoma[idx]]
                        
                        if bloque.profesor_id == profesor_id:
                            curso = self.cursos[bloque.curso_id]
                            fila.append(f"{curso.nombre}\n({AULAS[aula_idx]})")
                            clase_encontrada = True
                            break
                
                if not clase_encontrada:
                    fila.append("---")
            
            horario_matriz.append(fila)
        
        df = pd.DataFrame(
            horario_matriz,
            index=HORAS_PEDAGOGICAS,
            columns=DIAS
        )
        
        return df
    
    def ejecutar_algoritmo_genetico(self, verbose=True):
        """
        Ejecuta el algoritmo genético usando PyGAD
        """
        # Generar población inicial
        poblacion_inicial = []
        for _ in range(POBLACION_INICIAL):
            poblacion_inicial.append(self.crear_cromosoma_inicial())
        
        poblacion_inicial = np.array(poblacion_inicial)
        
        # Configurar PyGAD
        ga_instance = pygad.GA(
            num_generations=NUM_GENERACIONES,
            num_parents_mating=NUM_PADRES,
            fitness_func=self.calcular_fitness,
            sol_per_pop=POBLACION_INICIAL,
            num_genes=LONGITUD_CROMOSOMA,
            initial_population=poblacion_inicial,
            gene_type=int,
            gene_space=list(range(-1, len(self.bloques))),
            parent_selection_type="tournament",
            crossover_type="two_points",
            mutation_type="random",
            mutation_probability=TASA_MUTACION,
            crossover_probability=TASA_CRUCE,
            keep_elitism=2,
            suppress_warnings=True
        )
        
        # Callback para seguimiento del progreso
        def on_generation(ga_inst):
            if verbose and ga_inst.generations_completed % 50 == 0:
                mejor_fitness = ga_inst.best_solution()[1]
                print(f"Generación {ga_inst.generations_completed}: Fitness = {mejor_fitness:.2f}")
                self.historial_fitness.append(mejor_fitness)
        
        ga_instance.on_generation = on_generation
        
        # Ejecutar algoritmo
        if verbose:
            print("Iniciando algoritmo genético...")
            print(f"Población: {POBLACION_INICIAL}, Generaciones: {NUM_GENERACIONES}")
            print("-" * 50)
        
        ga_instance.run()
        
        # Obtener mejor solución
        self.mejor_cromosoma, self.mejor_fitness, _ = ga_instance.best_solution()
        self.mejor_cromosoma = self.mejor_cromosoma.astype(np.int32)
        
        if verbose:
            print("-" * 50)
            print(f"Algoritmo finalizado!")
            print(f"Mejor fitness: {self.mejor_fitness:.2f}")
        
        return self.mejor_cromosoma, self.mejor_fitness
    
    def mostrar_estadisticas(self, cromosoma: np.ndarray):
        """Muestra estadísticas del horario generado"""
        print("\n" + "=" * 60)
        print("ESTADÍSTICAS DEL HORARIO GENERADO")
        print("=" * 60)
        
        # Horas asignadas por profesor
        horas_asignadas = {prof_id: 0 for prof_id in self.profesores.keys()}
        bloques_contados = set()
        
        for idx in range(LONGITUD_CROMOSOMA):
            if cromosoma[idx] >= 0 and cromosoma[idx] < len(self.bloques):
                if cromosoma[idx] not in bloques_contados:
                    bloques_contados.add(cromosoma[idx])
                    bloque = self.bloques[cromosoma[idx]]
                    horas_asignadas[bloque.profesor_id] += bloque.duracion
        
        print("\nHoras semanales por profesor:")
        print("-" * 40)
        for prof_id, profesor in self.profesores.items():
            asignadas = horas_asignadas[prof_id]
            requeridas = profesor.horas_semanales
            estado = "OK" if asignadas == requeridas else "X"
            print(f"{profesor.nombre}: {asignadas}/{requeridas} horas [{estado}]")
        
        # Utilización de aulas
        print("\nUtilización de aulas:")
        print("-" * 40)
        for aula_idx, aula_nombre in enumerate(AULAS):
            horas_ocupadas = 0
            for i in range(POSICIONES_POR_AULA):
                idx = aula_idx * POSICIONES_POR_AULA + i
                if cromosoma[idx] >= 0:
                    horas_ocupadas += 1
            
            porcentaje = (horas_ocupadas / POSICIONES_POR_AULA) * 100
            print(f"{aula_nombre}: {horas_ocupadas}/{POSICIONES_POR_AULA} horas ({porcentaje:.1f}%)")
        
        # Horas vacías por profesor
        print("\nHoras vacías entre clases por profesor:")
        print("-" * 40)
        for profesor_id, profesor in self.profesores.items():
            total_horas_vacias = 0
            
            for dia in range(NUM_DIAS):
                horas_profesor = []
                
                for hora in range(HORAS_POR_DIA):
                    for aula in range(NUM_AULAS):
                        idx = self.indices_locales_a_global(aula, dia, hora)
                        if cromosoma[idx] >= 0 and cromosoma[idx] < len(self.bloques):
                            bloque = self.bloques[cromosoma[idx]]
                            if bloque.profesor_id == profesor_id:
                                horas_profesor.append(hora)
                                break
                
                if len(horas_profesor) > 1:
                    horas_profesor.sort()
                    for i in range(len(horas_profesor) - 1):
                        diferencia = horas_profesor[i + 1] - horas_profesor[i] - 1
                        total_horas_vacias += diferencia
            
            print(f"{profesor.nombre}: {total_horas_vacias} horas vacías")

# ========================= FUNCIONES DE EJEMPLO Y PRUEBA =========================

def crear_datos_ejemplo():
    """Crea datos de ejemplo para probar el sistema"""
    sistema = SistemaHorarios()
    
    # Crear profesores con disponibilidad
    profesores_data = [
        {
            "id": 1,
            "nombre": "Edwin",
            "codigo": "PROF001",
            "horas_semanales": 20,
            "disponibilidad": np.ones((NUM_DIAS, HORAS_POR_DIA))  # Disponible todas las horas
        },
        {
            "id": 2,
            "nombre": "Carlos",
            "codigo": "PROF002",
            "horas_semanales": 18,
            "disponibilidad": np.ones((NUM_DIAS, HORAS_POR_DIA))  # Disponible todas las horas
        },
        {
            "id": 3,
            "nombre": "María",
            "codigo": "PROF003",
            "horas_semanales": 15,
            "disponibilidad": np.ones((NUM_DIAS, HORAS_POR_DIA))
        },
        {
            "id": 4,
            "nombre": "Juan",
            "codigo": "PROF004",
            "horas_semanales": 12,
            "disponibilidad": np.ones((NUM_DIAS, HORAS_POR_DIA))
        }
    ]
    
    # Configurar disponibilidad específica (ejemplo: Carlos no disponible los lunes)
    profesores_data[1]["disponibilidad"][0, :] = 0  # Carlos no disponible los lunes
    
    # María solo disponible por las mañanas (primeras 4 horas)
    profesores_data[2]["disponibilidad"][:, 4:] = 0
    
    # Agregar profesores al sistema
    for prof_data in profesores_data:
        profesor = Profesor(
            id=prof_data["id"],
            nombre=prof_data["nombre"],
            codigo=prof_data["codigo"],
            disponibilidad=prof_data["disponibilidad"],
            horas_semanales=prof_data["horas_semanales"]
        )
        sistema.agregar_profesor(profesor)
    
    # Crear cursos
    cursos_data = [
        {"id": 1, "nombre": "Matemáticas", "codigo": "MAT101", "profesor_id": 1, "horas": 8},
        {"id": 2, "nombre": "Computación", "codigo": "COMP101", "profesor_id": 1, "horas": 6},
        {"id": 3, "nombre": "Física", "codigo": "FIS101", "profesor_id": 2, "horas": 8},
        {"id": 4, "nombre": "Química", "codigo": "QUI101", "profesor_id": 2, "horas": 6},
        {"id": 5, "nombre": "Historia", "codigo": "HIS101", "profesor_id": 3, "horas": 6},
        {"id": 6, "nombre": "Geografía", "codigo": "GEO101", "profesor_id": 3, "horas": 4},
        {"id": 7, "nombre": "Inglés", "codigo": "ING101", "profesor_id": 4, "horas": 6},
        {"id": 8, "nombre": "Literatura", "codigo": "LIT101", "profesor_id": 4, "horas": 4},
        {"id": 9, "nombre": "Arte", "codigo": "ART101", "profesor_id": 1, "horas": 4},
        {"id": 10, "nombre": "Educación Física", "codigo": "EF101", "profesor_id": 2, "horas": 4},
    ]
    
    for curso_data in cursos_data:
        curso = Curso(
            id=curso_data["id"],
            nombre=curso_data["nombre"],
            codigo=curso_data["codigo"],
            profesor_id=curso_data["profesor_id"],
            horas_semanales=curso_data["horas"]
        )
        sistema.agregar_curso(curso)
    
    return sistema

def main():
    """Función principal para ejecutar el sistema"""
    print("=" * 60)
    print("SISTEMA DE OPTIMIZACIÓN DE HORARIOS ESCOLARES")
    print("Usando Algoritmo Genético con PyGAD")
    print("=" * 60)
    
    # Crear sistema con datos de ejemplo
    sistema = crear_datos_ejemplo()
    
    print(f"\nProfesores registrados: {len(sistema.profesores)}")
    print(f"Cursos registrados: {len(sistema.cursos)}")
    
    # Generar bloques
    sistema.generar_bloques()
    print(f"Bloques generados: {len(sistema.bloques)}")
    
    # Ejecutar algoritmo genético
    print("\n" + "=" * 60)
    mejor_solucion, mejor_fitness = sistema.ejecutar_algoritmo_genetico(verbose=True)
    
    # Mostrar estadísticas
    sistema.mostrar_estadisticas(mejor_solucion)
    
    # Convertir a horarios legibles
    print("\n" + "=" * 60)
    print("HORARIOS POR AULA")
    print("=" * 60)
    
    horarios = sistema.cromosoma_a_horario(mejor_solucion)
    for aula_nombre, df_horario in horarios.items():
        print(f"\n{'-' * 40}")
        print(f"AULA: {aula_nombre}")
        print(f"{'-' * 40}")
        print(df_horario.to_string())
    
    # Mostrar horario de un profesor específico
    print("\n" + "=" * 60)
    print("HORARIO INDIVIDUAL - Profesor: Edwin")
    print("=" * 60)
    horario_edwin = sistema.horario_profesor(mejor_solucion, 1)
    print(horario_edwin.to_string())
    
    return sistema, mejor_solucion

if __name__ == "__main__":
    sistema, solucion = main()