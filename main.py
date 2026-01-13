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
PESO_CRUCES = 1000         # Penalización MUY ALTA por cruce de profesores (DURO)
PESO_DISPONIBILIDAD = 1000 # Penalización MUY ALTA por violar disponibilidad (DURO)
PESO_HORAS_VACIAS = 200    # Penalización ALTA por cada hora vacía
PESO_HORAS_SEMANALES = 50  # Penalización por no cumplir horas semanales

# Bonificaciones (refuerzo positivo)
BONUS_BLOQUES_CONSECUTIVOS = 25   # Bonificación por tener clases seguidas
BONUS_INICIO_TEMPRANO = 5         # Bonificación por empezar temprano
BONUS_COMPACTACION = 100          # Bonificación MUY ALTA por día sin huecos

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
        Crea un cromosoma inicial válido OPTIMIZADO para minimizar horas vacías.
        Prioriza asignar bloques del mismo profesor de forma consecutiva.
        """
        cromosoma = np.full(LONGITUD_CROMOSOMA, -1, dtype=np.int32)

        # Agrupar bloques por profesor para asignarlos juntos
        bloques_por_profesor = {}
        for bloque_id, bloque in enumerate(self.bloques):
            if bloque.profesor_id not in bloques_por_profesor:
                bloques_por_profesor[bloque.profesor_id] = []
            bloques_por_profesor[bloque.profesor_id].append(bloque_id)

        # Procesar cada profesor
        for profesor_id, bloque_ids in bloques_por_profesor.items():
            profesor = self.profesores[profesor_id]
            random.shuffle(bloque_ids)  # Aleatorizar orden de bloques

            for bloque_id in bloque_ids:
                bloque = self.bloques[bloque_id]

                # Buscar posiciones válidas con PRIORIDAD a posiciones consecutivas
                posiciones_validas = []
                posiciones_prioritarias = []  # Adyacentes a clases existentes

                # Obtener curso y aula preferida
                curso = self.cursos[bloque.curso_id]
                aulas_permitidas = []

                # Si el curso tiene aula preferida, solo usar esa aula
                if curso.aula_preferida:
                    try:
                        aula_idx = AULAS.index(curso.aula_preferida)
                        aulas_permitidas = [aula_idx]
                    except ValueError:
                        aulas_permitidas = list(range(NUM_AULAS))
                else:
                    aulas_permitidas = list(range(NUM_AULAS))

                for aula in aulas_permitidas:
                    for dia in range(NUM_DIAS):
                        for hora in range(HORAS_POR_DIA - bloque.duracion + 1):
                            # Verificar disponibilidad del profesor
                            disponible = True
                            for h in range(bloque.duracion):
                                if profesor.disponibilidad[dia, hora + h] == 0:
                                    disponible = False
                                    break

                            if not disponible:
                                continue

                            # Verificar que las posiciones estén libres
                            posiciones_libres = True
                            for h in range(bloque.duracion):
                                idx = self.indices_locales_a_global(aula, dia, hora + h)
                                if cromosoma[idx] != -1:
                                    posiciones_libres = False
                                    break

                            if not posiciones_libres:
                                continue

                            # Verificar si es ADYACENTE a otra clase del mismo profesor
                            es_adyacente = False
                            # Verificar hora anterior
                            if hora > 0:
                                for a in range(NUM_AULAS):
                                    idx_prev = self.indices_locales_a_global(a, dia, hora - 1)
                                    if cromosoma[idx_prev] >= 0:
                                        bloque_prev = self.bloques[cromosoma[idx_prev]]
                                        if bloque_prev.profesor_id == profesor_id:
                                            es_adyacente = True
                                            break
                            # Verificar hora posterior
                            if not es_adyacente and hora + bloque.duracion < HORAS_POR_DIA:
                                for a in range(NUM_AULAS):
                                    idx_next = self.indices_locales_a_global(a, dia, hora + bloque.duracion)
                                    if cromosoma[idx_next] >= 0:
                                        bloque_next = self.bloques[cromosoma[idx_next]]
                                        if bloque_next.profesor_id == profesor_id:
                                            es_adyacente = True
                                            break

                            if es_adyacente:
                                posiciones_prioritarias.append((aula, dia, hora))
                            else:
                                posiciones_validas.append((aula, dia, hora))

                # Elegir posición: prioritarias primero, luego válidas
                if posiciones_prioritarias:
                    aula, dia, hora = random.choice(posiciones_prioritarias)
                elif posiciones_validas:
                    aula, dia, hora = random.choice(posiciones_validas)
                else:
                    continue  # No hay posición válida

                # Asignar bloque
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

        # === CÁLCULO DE BONIFICACIONES ===
        bonificacion_total = 0

        # 5. Bonificación por bloques CONSECUTIVOS del mismo profesor
        for profesor_id in self.profesores.keys():
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

                # Contar bloques consecutivos
                if len(horas_profesor) > 1:
                    horas_profesor.sort()
                    consecutivos = 0
                    for i in range(len(horas_profesor) - 1):
                        if horas_profesor[i + 1] - horas_profesor[i] == 1:
                            consecutivos += 1
                    bonificacion_total += consecutivos * BONUS_BLOQUES_CONSECUTIVOS

        # 6. Bonificación por empezar TEMPRANO (primeras 4 horas)
        for profesor_id in self.profesores.keys():
            for dia in range(NUM_DIAS):
                for hora in range(4):  # Primeras 4 horas
                    for aula in range(NUM_AULAS):
                        idx = self.indices_locales_a_global(aula, dia, hora)
                        if cromosoma[idx] >= 0 and cromosoma[idx] < len(self.bloques):
                            bloque = self.bloques[cromosoma[idx]]
                            if bloque.profesor_id == profesor_id:
                                bonificacion_total += BONUS_INICIO_TEMPRANO
                                break

        # 7. Bonificación por COMPACTACIÓN (todas las clases seguidas sin huecos)
        for profesor_id in self.profesores.keys():
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

                # Si tiene clases y están perfectamente compactadas
                if len(horas_profesor) > 1:
                    horas_profesor.sort()
                    rango_esperado = horas_profesor[-1] - horas_profesor[0] + 1
                    if len(horas_profesor) == rango_esperado:
                        # ¡Perfecto! Sin huecos
                        bonificacion_total += BONUS_COMPACTACION * len(horas_profesor)

        # === CÁLCULO DE FITNESS ===

        # FASE 1: Verificar restricciones DURAS
        restricciones_duras = num_cruces + num_violaciones_disponibilidad

        if restricciones_duras > 0:
            # Si hay violaciones duras, penalizar pero permitir evolución
            penalizacion_duras = restricciones_duras * 500
            penalizacion_vacias_suave = total_horas_vacias * 10
            penalizacion_horas_suave = diferencia_horas_semanales * 5

            fitness = 10000 - penalizacion_duras - penalizacion_vacias_suave - penalizacion_horas_suave
            # Agregar bonificaciones reducidas para guiar evolución
            fitness += bonificacion_total * 0.1
            return max(1, fitness)

        # FASE 2: Sin violaciones duras, optimizar restricciones BLANDAS
        # Penalización DIRECTA por cada hora vacía
        penalizacion_vacias = total_horas_vacias * PESO_HORAS_VACIAS

        # Penalización por horas semanales no cumplidas
        penalizacion_horas = diferencia_horas_semanales * PESO_HORAS_SEMANALES

        # Calcular fitness final
        penalizacion_total = penalizacion_vacias + penalizacion_horas

        # Fitness = base + bonificaciones - penalizaciones
        fitness = 10000 + bonificacion_total - penalizacion_total

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
        
        # Configurar PyGAD con parámetros OPTIMIZADOS
        ga_instance = pygad.GA(
            num_generations=NUM_GENERACIONES,
            num_parents_mating=NUM_PADRES,
            fitness_func=self.calcular_fitness,
            sol_per_pop=POBLACION_INICIAL,
            num_genes=LONGITUD_CROMOSOMA,
            initial_population=poblacion_inicial,
            gene_type=int,
            gene_space=list(range(-1, len(self.bloques))),
            # Selección por torneo con mayor presión selectiva
            parent_selection_type="tournament",
            K_tournament=5,  # Mayor presión selectiva
            # Cruce uniforme para mejor mezcla de genes
            crossover_type="uniform",
            crossover_probability=TASA_CRUCE,
            # Mutación adaptativa
            mutation_type="random",
            mutation_probability=TASA_MUTACION,
            mutation_percent_genes="default",  # Usar valor por defecto de PyGAD
            # Elitismo aumentado para preservar mejores soluciones
            keep_elitism=5,
            # Permitir duplicados para mantener diversidad
            allow_duplicate_genes=True,
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
        {"id": 1, "nombre": "Dalila Luz Sánchez Casiano", "codigo": "PROF001", "horas_semanales": 7, "disponibilidad": np.ones((NUM_DIAS, HORAS_POR_DIA))},
        {"id": 2, "nombre": "Jorge Ruiz Campos", "codigo": "PROF002", "horas_semanales": 3, "disponibilidad": np.ones((NUM_DIAS, HORAS_POR_DIA))},
        {"id": 3, "nombre": "Ana Rosa Guevara Rodríguez", "codigo": "PROF003", "horas_semanales": 8, "disponibilidad": np.ones((NUM_DIAS, HORAS_POR_DIA))},
        {"id": 4, "nombre": "Leonardo Hernández Cruzado", "codigo": "PROF004", "horas_semanales": 6, "disponibilidad": np.ones((NUM_DIAS, HORAS_POR_DIA))},
        {"id": 5, "nombre": "Elia Cayetano Avalos", "codigo": "PROF005", "horas_semanales": 11, "disponibilidad": np.ones((NUM_DIAS, HORAS_POR_DIA))},
        {"id": 6, "nombre": "Julio Cesar Chaupe Cruz", "codigo": "PROF006", "horas_semanales": 9, "disponibilidad": np.ones((NUM_DIAS, HORAS_POR_DIA))},
        {"id": 7, "nombre": "Víctor Minchola Achín", "codigo": "PROF007", "horas_semanales": 9, "disponibilidad": np.ones((NUM_DIAS, HORAS_POR_DIA))},
        {"id": 8, "nombre": "María Paisig Gurreonero", "codigo": "PROF008", "horas_semanales": 3, "disponibilidad": np.ones((NUM_DIAS, HORAS_POR_DIA))},
        {"id": 9, "nombre": "Margarita Rivera Paredes", "codigo": "PROF009", "horas_semanales": 3, "disponibilidad": np.ones((NUM_DIAS, HORAS_POR_DIA))},
        {"id": 10, "nombre": "Saira Cuba Ruiz", "codigo": "PROF010", "horas_semanales": 4, "disponibilidad": np.ones((NUM_DIAS, HORAS_POR_DIA))},
        {"id": 11, "nombre": "Joel Montenegro", "codigo": "PROF011", "horas_semanales": 5, "disponibilidad": np.ones((NUM_DIAS, HORAS_POR_DIA))},
        {"id": 12, "nombre": "Contrato de Arte", "codigo": "PROF012", "horas_semanales": 4, "disponibilidad": np.ones((NUM_DIAS, HORAS_POR_DIA))},
        {"id": 13, "nombre": "Orlando Mendoza Chumpitazi", "codigo": "PROF013", "horas_semanales": 3, "disponibilidad": np.ones((NUM_DIAS, HORAS_POR_DIA))}
    ]
    
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
    
    # Crear cursos CON AULA ASIGNADA - Datos reales
    cursos_data = [
        {"id": 1, "nombre": "Matemática", "codigo": "MAT3A", "profesor_id": 1, "horas": 3, "aula_preferida": "3A"},
        {"id": 2, "nombre": "Matemática", "codigo": "MAT3B", "profesor_id": 1, "horas": 4, "aula_preferida": "3B"},
        {"id": 3, "nombre": "Matemática", "codigo": "MAT4", "profesor_id": 8, "horas": 3, "aula_preferida": "4"},
        {"id": 4, "nombre": "Comunicación", "codigo": "COM3A", "profesor_id": 3, "horas": 4, "aula_preferida": "3A"},
        {"id": 5, "nombre": "Comunicación", "codigo": "COM3B", "profesor_id": 3, "horas": 4, "aula_preferida": "3B"},
        {"id": 6, "nombre": "Comunicación", "codigo": "COM4", "profesor_id": 9, "horas": 3, "aula_preferida": "4"},
        {"id": 7, "nombre": "CyT", "codigo": "CYT3A", "profesor_id": 5, "horas": 3, "aula_preferida": "3A"},
        {"id": 8, "nombre": "Tutoría", "codigo": "TUT3A", "profesor_id": 5, "horas": 2, "aula_preferida": "3A"},
        {"id": 9, "nombre": "CyT", "codigo": "CYT3B", "profesor_id": 5, "horas": 2, "aula_preferida": "3B"},
        {"id": 10, "nombre": "CyT", "codigo": "CYT4", "profesor_id": 5, "horas": 4, "aula_preferida": "4"},
        {"id": 11, "nombre": "Inglés", "codigo": "ING3A", "profesor_id": 7, "horas": 3, "aula_preferida": "3A"},
        {"id": 12, "nombre": "Inglés", "codigo": "ING3B", "profesor_id": 7, "horas": 3, "aula_preferida": "3B"},
        {"id": 13, "nombre": "Inglés", "codigo": "ING4", "profesor_id": 7, "horas": 3, "aula_preferida": "4"},
        {"id": 14, "nombre": "Educación Física", "codigo": "EF3A", "profesor_id": 6, "horas": 2, "aula_preferida": "3A"},
        {"id": 15, "nombre": "Educación Física", "codigo": "EF3B", "profesor_id": 6, "horas": 2, "aula_preferida": "3B"},
        {"id": 16, "nombre": "Arte y Cultura", "codigo": "AYC3B", "profesor_id": 6, "horas": 2, "aula_preferida": "3B"},
        {"id": 17, "nombre": "Tutoría", "codigo": "TUT3B", "profesor_id": 6, "horas": 1, "aula_preferida": "3B"},
        {"id": 18, "nombre": "Educación Física", "codigo": "EF4", "profesor_id": 6, "horas": 2, "aula_preferida": "4"},
        {"id": 19, "nombre": "DPCC", "codigo": "DPCC3A", "profesor_id": 4, "horas": 2, "aula_preferida": "3A"},
        {"id": 20, "nombre": "DPCC", "codigo": "DPCC3B", "profesor_id": 4, "horas": 2, "aula_preferida": "3B"},
        {"id": 21, "nombre": "DPCC", "codigo": "DPCC4", "profesor_id": 4, "horas": 2, "aula_preferida": "4"},
        {"id": 22, "nombre": "CCSS", "codigo": "CCSS3A", "profesor_id": 10, "horas": 2, "aula_preferida": "3A"},
        {"id": 23, "nombre": "CCSS", "codigo": "CCSS3B", "profesor_id": 10, "horas": 2, "aula_preferida": "3B"},
        {"id": 24, "nombre": "CCSS", "codigo": "CCSS4", "profesor_id": 2, "horas": 2, "aula_preferida": "4"},
        {"id": 25, "nombre": "Tutoría", "codigo": "TUT4", "profesor_id": 2, "horas": 1, "aula_preferida": "4"},
        {"id": 26, "nombre": "EPT", "codigo": "EPT3A", "profesor_id": 11, "horas": 2, "aula_preferida": "3A"},
        {"id": 27, "nombre": "EPT", "codigo": "EPT3B", "profesor_id": 11, "horas": 1, "aula_preferida": "3B"},
        {"id": 28, "nombre": "EPT", "codigo": "EPT4", "profesor_id": 11, "horas": 2, "aula_preferida": "4"},
        {"id": 29, "nombre": "Arte y Cultura", "codigo": "AYC3A", "profesor_id": 12, "horas": 2, "aula_preferida": "3A"},
        {"id": 30, "nombre": "Arte y Cultura", "codigo": "AYC4", "profesor_id": 12, "horas": 2, "aula_preferida": "4"},
        {"id": 31, "nombre": "Religión", "codigo": "REL3A", "profesor_id": 13, "horas": 1, "aula_preferida": "3A"},
        {"id": 32, "nombre": "Religión", "codigo": "REL3B", "profesor_id": 13, "horas": 1, "aula_preferida": "3B"},
        {"id": 33, "nombre": "Religión", "codigo": "REL4", "profesor_id": 13, "horas": 1, "aula_preferida": "4"}
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
    
    # Mostrar horario individual de TODOS los profesores
    for prof_id, profesor in sistema.profesores.items():
        print("\n" + "=" * 60)
        print(f"HORARIO INDIVIDUAL - Profesor: {profesor.nombre}")
        print("=" * 60)
        horario_prof = sistema.horario_profesor(mejor_solucion, prof_id)
        print(horario_prof.to_string())
    
    return sistema, mejor_solucion

if __name__ == "__main__":
    sistema, solucion = main()