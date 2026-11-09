"""Автопилот выведения Luna-16 Mun Carrier V2 Fixed/V3 на орбиту Кербина.

Сценарий рассчитан на проверенную силовую схему носителя V2 Fixed/V3:

1. вертикальный набор высоты;
2. очень плавный гравитационный разворот на восток;
3. отделение боковых ТТУ после выгорания;
4. переход с центральной ступени на орбитальную;
5. формирование апоцентра и круговой орбиты;
6. запись телеметрии в CSV.

Перед запуском откройте корабль на стартовом столе, включите сервер kRPC и
запустите этот файл из PyCharm. Для реального старта потребуется ввести ПУСК.
"""

from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import krpc


# ------------------------------ Настройки ---------------------------------

EXPECTED_VESSEL_NAMES = (
    "Luna-16 Mun Carrier V2 Fixed",
    "Luna-16 Mun Carrier V3",
    "Luna-16 Mun Carrier V3 Mission Plus",
    "Luna-16 Mun Carrier V3 Landing Heavy",
    "Luna-16 Mun Carrier V3 Mission Heavy",
    "Luna-16 Mun Carrier V3 Fixed 1",
    "Luna-16 Mun Carrier V3 Fixed 2",
    "Luna-16 Mun Carrier V3 Fixed 3",
)

TARGET_ORBIT_ALTITUDE = 100_000.0       # круговая орбита 100 км
APOAPSIS_CUTOFF_MARGIN = 1_500.0         # не дотягивать апоцентр ценой его прохода
HEADING = 90.0                           # строго на восток
VERTICAL_ASCENT_END = 2_500.0            # до 2,5 км летим вертикально
BOOSTER_SEPARATION_FUEL = 1.0            # остаток SolidFuel, ед.
MAX_DYNAMIC_PRESSURE = 30_000.0          # ограничитель Max-Q, Па
MIN_THROTTLE_IN_ATMOSPHERE = 0.45
TELEMETRY_PERIOD = 0.20                  # 5 измерений в секунду
UPPER_STAGE_SEPARATION_DELAY = 0.2       # минимальная пауза на отход ступени
UPPER_STAGE_IGNITION_RAMP = 0.25         # почти мгновенный набор тяги Poodle
MIN_SAFE_UPPER_STAGE_ALTITUDE = 76_000.0
MAX_SAFE_DESCENT_SPEED = -40.0

G0 = 9.80665


@dataclass
class FlightState:
    """Служебное состояние ступеней носителя."""

    boosters_separated: bool = False
    fairing_jettisoned: bool = False
    upper_stage_activated: bool = False


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def lerp(x: float, x0: float, x1: float, y0: float, y1: float) -> float:
    """Линейная интерполяция с ограничением по краям интервала."""
    if x1 == x0:
        return y1
    k = clamp((x - x0) / (x1 - x0), 0.0, 1.0)
    return y0 + k * (y1 - y0)


def commanded_pitch(altitude: float) -> float:
    """Мягкая программа тангажа для длинного и чувствительного носителя.

    Самый плотный участок атмосферы ракета проходит почти вертикально. После
    10 км разворот ускоряется, но соседние участки стыкуются без скачков.
    """
    schedule = (
        (0.0, 90.0),
        (VERTICAL_ASCENT_END, 90.0),
        (6_000.0, 87.0),
        (10_000.0, 82.0),
        (15_000.0, 70.0),
        (25_000.0, 45.0),
        (40_000.0, 25.0),
        (55_000.0, 10.0),
        (70_000.0, 3.0),
    )

    for (h0, p0), (h1, p1) in zip(schedule, schedule[1:]):
        if altitude <= h1:
            return lerp(altitude, h0, h1, p0, p1)
    return schedule[-1][1]


def atmospheric_throttle(altitude: float, dynamic_pressure: float) -> float:
    """Ограничивает жидкостный двигатель около Max-Q.

    Твердотопливные ускорители не дросселируются, поэтому регулятор не пытается
    полностью закрыть тягу и сохраняет управляемость центральным двигателем.
    """
    if altitude < 8_000.0 or dynamic_pressure <= MAX_DYNAMIC_PRESSURE:
        return 1.0

    ratio = MAX_DYNAMIC_PRESSURE / max(dynamic_pressure, 1.0)
    return clamp(ratio, MIN_THROTTLE_IN_ATMOSPHERE, 1.0)


def engines_with_name(vessel, name_fragment: str):
    fragment = name_fragment.lower()
    return [
        engine
        for engine in vessel.parts.engines
        if fragment in engine.part.name.lower()
    ]


def resource_amount(parts, resource_name: str) -> float:
    total = 0.0
    for part in parts:
        try:
            if part.resources.has_resource(resource_name):
                total += part.resources.amount(resource_name)
        except RuntimeError:
            # После отделения объект детали перестаёт принадлежать активному
            # кораблю. Это нормальная ситуация и не является аварией.
            pass
    return total


def validate_vehicle(vessel):
    boosters = engines_with_name(vessel, "MassiveBooster")
    core_engines = engines_with_name(vessel, "liquidEngineMainsail")
    upper_engines = engines_with_name(vessel, "liquidEngine2-2")

    errors = []
    if not any(name.lower() in vessel.name.lower() for name in EXPECTED_VESSEL_NAMES):
        # KSP иногда сохраняет имя из старого ModuleVesselNaming (например,
        # ``Ariane 5``), хотя загружен правильный .craft. Поэтому имя — только
        # предупреждение; решение принимается по реальному составу двигателей.
        print(
            f"Предупреждение: внутреннее имя корабля — '{vessel.name}'. "
            "Проверяю конструкцию по двигателям."
        )
    if len(boosters) not in {2, 4}:
        errors.append(
            f"найдено ТТУ: {len(boosters)}, ожидалось: 2 или 4"
        )
    if len(core_engines) != 1:
        errors.append(
            f"центральных Mainsail найдено: {len(core_engines)}, ожидался: 1"
        )
    if len(upper_engines) != 1:
        errors.append(
            f"орбитальных Poodle найдено: {len(upper_engines)}, ожидался: 1"
        )

    if errors:
        raise RuntimeError("Проверка ракеты не пройдена:\n- " + "\n- ".join(errors))

    return boosters, core_engines, upper_engines


def jettison_fairing(vessel, state: FlightState) -> None:
    """Раскрывает только процедурный обтекатель, не переключая ступень.

    Прямой вызов события безопаснее ``activate_next_stage()``: он не может
    случайно задействовать разделитель посадочного аппарата или другую деталь,
    оказавшуюся в соседней позиции стека ступеней.
    """
    if state.fairing_jettisoned:
        return

    fairing_modules = [
        module
        for part in vessel.parts.all
        for module in part.modules
        if module.name == "ModuleProceduralFairing"
    ]
    if not fairing_modules:
        print("Обтекатель уже сброшен или его модуль не найден.")
        state.fairing_jettisoned = True
        return

    for module in fairing_modules:
        deploy_event = next(
            (event for event in module.event_list if event.name == "DeployFairing"),
            None,
        )
        if deploy_event is None:
            raise RuntimeError("Событие раскрытия обтекателя не найдено.")
        if not deploy_event.active:
            print("Обтекатель уже был сброшен вручную.")
            state.fairing_jettisoned = True
            return
        try:
            deploy_event.trigger()
        except RuntimeError as error:
            raise RuntimeError(
                "Не удалось безопасно раскрыть обтекатель."
            ) from error

    state.fairing_jettisoned = True
    print("Обтекатель сброшен вместе с центральной ступенью.")


def wait_for_active_vessel(conn, timeout: float = 15.0):
    """Пережидает короткое состояние загрузки KSP после разделения ступени."""
    deadline = time.monotonic() + timeout
    last_error = None
    stable_reads = 0
    candidate = None
    while time.monotonic() < deadline:
        try:
            candidate = conn.space_center.active_vessel
            if candidate is None:
                stable_reads = 0
                time.sleep(0.1)
                continue
            # Простого получения active_vessel недостаточно: во время смены
            # состояния RPC-объект уже существует, но его Orbit ещё не загружен.
            _ = candidate.orbit.apoapsis_altitude
            _ = candidate.mass
            stable_reads += 1
            if stable_reads >= 3:
                return candidate
        except Exception as error:
            if error.__class__.__name__ != "ObjectNotLoadedException":
                raise
            last_error = error
            stable_reads = 0
        time.sleep(0.15)
    raise RuntimeError(
        "KSP не завершил переключение на новую ступень за 15 секунд."
    ) from last_error


def activate_upper_stage(conn, vessel, state: FlightState):
    """Отделяет пустой центральный блок и запускает Poodle.

    Газ сначала полностью закрывается, чтобы струя Poodle не ударила по только
    что отделённой центральной ступени. После паузы двигатель плавно набирает
    тягу. Разрешены не более двух переключений для совместимости с V2 и V3.
    """
    if state.upper_stage_activated:
        return vessel

    print("Центральная ступень выработана. Переход на орбитальную ступень...")
    vessel.control.throttle = 0.0

    for _ in range(2):
        vessel.control.activate_next_stage()
        time.sleep(0.8)
        # После срабатывания разделителя активный объект может получить новый
        # RPC-идентификатор, поэтому обязательно запрашиваем его заново.
        vessel = wait_for_active_vessel(conn)
        upper_engines = engines_with_name(vessel, "liquidEngine2-2")
        if any(engine.active and engine.has_fuel for engine in upper_engines):
            # Обтекатель снимаем адресно сразу после разделения. Он больше не
            # нужен вне плотной атмосферы, а солнечные панели должны быть
            # открыты до длительного перелёта к Муне.
            jettison_fairing(vessel, state)
            print(
                f"Центральная ступень отделена. Пауза "
                f"{UPPER_STAGE_SEPARATION_DELAY:.1f} с перед зажиганием."
            )
            vessel.control.throttle = 0.0
            time.sleep(UPPER_STAGE_SEPARATION_DELAY)

            # После разделения повторно получаем активный объект и двигатель.
            vessel = wait_for_active_vessel(conn)
            upper_engines = engines_with_name(vessel, "liquidEngine2-2")
            fueled_engines = [engine for engine in upper_engines if engine.has_fuel]
            if not fueled_engines:
                raise RuntimeError("После разделения Poodle не имеет топлива.")
            for engine in fueled_engines:
                if not engine.active:
                    engine.active = True

            ramp_steps = 20
            for step in range(1, ramp_steps + 1):
                vessel.control.throttle = step / ramp_steps
                time.sleep(UPPER_STAGE_IGNITION_RAMP / ramp_steps)

            state.upper_stage_activated = True
            print("Орбитальная ступень Poodle плавно запущена.")
            return wait_for_active_vessel(conn)

    vessel.control.throttle = 0.0
    raise RuntimeError(
        "Poodle не запустился после двух безопасных переключений. "
        "Дальнейшее переключение запрещено, чтобы не затронуть посадочный модуль."
    )


def stage_if_needed(conn, vessel, state: FlightState, booster_parts):
    """Отделяет ТТУ и при необходимости запускает орбитальную ступень."""
    control = vessel.control
    if not state.boosters_separated:
        solid_fuel = resource_amount(booster_parts, "SolidFuel")
        if solid_fuel <= BOOSTER_SEPARATION_FUEL:
            print("ТТУ выработаны — отделение боковых ускорителей.")
            control.activate_next_stage()
            state.boosters_separated = True
            time.sleep(0.8)

    if state.boosters_separated and not state.upper_stage_activated:
        core_engines = engines_with_name(vessel, "liquidEngineMainsail")
        core_has_fuel = any(engine.has_fuel for engine in core_engines)
        if core_engines and not core_has_fuel:
            return activate_upper_stage(conn, vessel, state), True

    return vessel, False


def estimate_burn_time(vessel, delta_v: float) -> float:
    thrust = max(vessel.available_thrust, vessel.max_thrust)
    isp = vessel.specific_impulse
    if thrust <= 0.0 or isp <= 0.0:
        raise RuntimeError("Нет доступной тяги для расчёта импульса.")

    exhaust_velocity = isp * G0
    initial_mass = vessel.mass
    final_mass = initial_mass / math.exp(delta_v / exhaust_velocity)
    mass_flow = thrust / exhaust_velocity
    return (initial_mass - final_mass) / mass_flow


def circularization_delta_v(vessel) -> float:
    orbit = vessel.orbit
    body = orbit.body
    mu = body.gravitational_parameter
    radius = body.equatorial_radius
    apoapsis_radius = radius + orbit.apoapsis_altitude
    semi_major_axis = orbit.semi_major_axis

    velocity_at_apoapsis = math.sqrt(
        mu * (2.0 / apoapsis_radius - 1.0 / semi_major_axis)
    )
    circular_velocity = math.sqrt(mu / apoapsis_radius)
    return max(0.0, circular_velocity - velocity_at_apoapsis)


def wait_for_circularization_thrust(
    vessel,
    control,
    state: FlightState,
    timeout: float = 3.0,
) -> None:
    """Запускает только двигатель реально активной ступени.

    Пока центральная ступень не отделена, Poodle принудительно выключен, а
    циркуляризацию начинает Mainsail. Это не позволяет включить Poodle внутри
    ещё присоединённой центральной ступени.
    """
    if not state.upper_stage_activated:
        upper_engines = engines_with_name(vessel, "liquidEngine2-2")
        for engine in upper_engines:
            if engine.active:
                engine.active = False

        engines = engines_with_name(vessel, "liquidEngineMainsail")
        fueled_engines = [engine for engine in engines if engine.has_fuel]
        engine_name = "Mainsail"
        if not fueled_engines:
            control.throttle = 0.0
            raise RuntimeError(
                "Mainsail уже пуст, но центральная ступень ещё не отделена. "
                "Запуск Poodle заблокирован."
            )
    else:
        # После разделения Mainsail вообще не должен принадлежать активному
        # кораблю. Дополнительная проверка защищает от ошибочного выбора обломка.
        if engines_with_name(vessel, "liquidEngineMainsail"):
            control.throttle = 0.0
            raise RuntimeError(
                "После отделения активным остался объект с Mainsail; "
                "запуск Poodle заблокирован."
            )
        engines = engines_with_name(vessel, "liquidEngine2-2")
        fueled_engines = [engine for engine in engines if engine.has_fuel]
        engine_name = "Poodle"
        if not fueled_engines:
            raise RuntimeError("В орбитальной ступени Poodle закончилось топливо.")

    for engine in fueled_engines:
        if not engine.active:
            engine.active = True

    control.throttle = 1.0
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if vessel.thrust >= 1.0:
            print(
                f"{engine_name} подтвердил тягу: "
                f"{vessel.thrust / 1000.0:.1f} кН."
            )
            return
        time.sleep(0.05)

    statuses = ", ".join(
        f"active={engine.active}, has_fuel={engine.has_fuel}"
        for engine in engines
    )
    control.throttle = 0.0
    raise RuntimeError(
        f"{engine_name} не создал тягу за {timeout:.0f} с ({statuses})."
    )


def execute_node(conn, vessel, node, state: FlightState):
    control = vessel.control
    autopilot = vessel.auto_pilot

    delta_v = abs(node.prograde)
    burn_time = estimate_burn_time(vessel, delta_v)
    if not state.upper_stage_activated:
        # Если Mainsail закончится посреди импульса, остаток будет выполнять
        # менее тяговитый Poodle. Начинаем немного раньше, чтобы не пройти
        # апоцентр во время безопасного разделения ступеней.
        upper_engines = engines_with_name(vessel, "liquidEngine2-2")
        upper_thrust = sum(engine.max_vacuum_thrust for engine in upper_engines)
        if upper_thrust > 0.0:
            conservative_time = delta_v * vessel.mass / upper_thrust
            burn_time = max(burn_time, conservative_time)
    burn_start_ut = node.ut - burn_time / 2.0

    print(f"Расчётный импульс: {delta_v:.1f} м/с, время: {burn_time:.1f} с")
    autopilot.reference_frame = node.reference_frame
    autopilot.target_direction = (0.0, 1.0, 0.0)
    # В kRPC 0.7.0 автопилот включается через изменяемое свойство.
    autopilot.engaged = True
    autopilot.wait()


    # Оставляем запас времени после физического варпа для стабилизации аппарата.
    conn.space_center.warp_to(max(conn.space_center.ut, burn_start_ut - 8.0))
    while conn.space_center.ut < burn_start_ut:
        time.sleep(0.05)

    print("Начало циркуляризационного импульса.")
    if not state.upper_stage_activated:
        core = engines_with_name(vessel, "liquidEngineMainsail")
        if core and not any(engine.has_fuel for engine in core):
            print(
                "Mainsail закончился перед циркуляризацией — "
                "сначала отделяю центральную ступень."
            )
            vessel = activate_upper_stage(conn, vessel, state)
            control = vessel.control
            autopilot = vessel.auto_pilot
            nodes = list(control.nodes)
            if len(nodes) != 1:
                control.throttle = 0.0
                raise RuntimeError(
                    "После отделения центральной ступени манёвренный "
                    f"узел найден неоднозначно: {len(nodes)}."
                )
            node = nodes[0]
            autopilot.reference_frame = node.reference_frame
            autopilot.target_direction = (0.0, 1.0, 0.0)
            autopilot.engaged = True
            autopilot.wait()
    wait_for_circularization_thrust(vessel, control, state)
    previous_remaining = float("inf")
    increasing_samples = 0
    zero_thrust_since = None

    while True:
        remaining_vector = node.remaining_burn_vector(node.reference_frame)
        remaining = math.sqrt(sum(component**2 for component in remaining_vector))

        # Тонкое дросселирование в конце уменьшает перелёт через расчётный импульс.
        control.throttle = clamp(remaining / 25.0, 0.08, 1.0)

        # Если нужный импульс уже набран одновременно с исчерпанием Mainsail,
        # не отделяем ступень и не запускаем Poodle без необходимости.
        if remaining < 0.5:
            break

        if vessel.thrust < 1.0:
            if zero_thrust_since is None:
                zero_thrust_since = time.monotonic()
            if time.monotonic() - zero_thrust_since < 1.0:
                # Один-два нулевых отсчёта возможны при обновлении физики KSP.
                time.sleep(0.05)
                continue

            core = engines_with_name(vessel, "liquidEngineMainsail")
            core_empty = core and not any(engine.has_fuel for engine in core)
            control.throttle = 0.0
            if core_empty and not state.upper_stage_activated:
                print(
                    "Mainsail выработан во время циркуляризации — "
                    "безопасный переход на Poodle."
                )
                vessel = activate_upper_stage(conn, vessel, state)
                control = vessel.control
                autopilot = vessel.auto_pilot
                nodes = list(control.nodes)
                if len(nodes) != 1:
                    control.throttle = 0.0
                    raise RuntimeError(
                        "После отделения центральной ступени манёвренный "
                        f"узел найден неоднозначно: {len(nodes)}."
                    )
                node = nodes[0]
                autopilot.reference_frame = node.reference_frame
                autopilot.target_direction = (0.0, 1.0, 0.0)
                autopilot.engaged = True
                autopilot.wait()
                wait_for_circularization_thrust(vessel, control, state)
                zero_thrust_since = None
                previous_remaining = float("inf")
                increasing_samples = 0
                continue
            raise RuntimeError("Во время импульса пропала тяга.")
        else:
            zero_thrust_since = None

        if remaining > previous_remaining + 0.15:
            increasing_samples += 1
        else:
            increasing_samples = 0
        if increasing_samples >= 4:
            # Мы уже прошли минимум остаточного вектора.
            break

        previous_remaining = remaining
        time.sleep(0.05)

    control.throttle = 0.0
    node.remove()
    autopilot.engaged = False
    control.sas = True
    print("Циркуляризационный импульс завершён.")
    return vessel


def telemetry_path() -> Path:
    directory = Path(__file__).resolve().parent / "telemetry"
    directory.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return directory / f"ascent_{stamp}.csv"


def emergency_stop() -> None:
    """По возможности закрывает тягу при Ctrl+C или программной ошибке."""
    try:
        emergency_conn = krpc.connect(name="Luna-16 emergency stop")
        vessel = emergency_conn.space_center.active_vessel
        if vessel is not None:
            vessel.control.throttle = 0.0
            vessel.auto_pilot.engaged = False
            vessel.control.sas = True
        emergency_conn.close()
    except Exception:
        # Исходная ошибка важнее возможной ошибки повторного подключения.
        pass


def run() -> None:
    conn = krpc.connect(name="Luna-16 ascent autopilot")
    vessel = conn.space_center.active_vessel
    if vessel is None:
        conn.close()
        raise RuntimeError("Активный корабль не найден. Перейдите на стартовый стол.")

    boosters, _, _ = validate_vehicle(vessel)
    booster_parts = [engine.part for engine in boosters]
    state = FlightState()
    # В surface_reference_frame начало координат движется вместе с аппаратом,
    # поэтому скорости получались нулевыми. Система тела даёт реальную скорость.
    flight = vessel.flight(vessel.orbit.body.reference_frame)
    control = vessel.control
    autopilot = vessel.auto_pilot

    print(f"Корабль: {vessel.name}")
    # kRPC возвращает массу в килограммах; для вывода переводим её в тонны.
    print(f"Масса на старте: {vessel.mass / 1000.0:.2f} т")
    print(f"Цель: круговая орбита {TARGET_ORBIT_ALTITUDE / 1000:.0f} км")
    print("Проверьте: аппарат стоит неподвижно, тяга закрыта, kRPC подключён.")
    if input("Для запуска введите ПУСК: ").strip().upper() != "ПУСК":
        conn.close()
        print("Запуск отменён.")
        return

    log_path = telemetry_path()
    start_ut = conn.space_center.ut
    last_log_time = -TELEMETRY_PERIOD

    with log_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "mission_time_s",
                "altitude_m",
                "speed_m_s",
                "vertical_speed_m_s",
                "horizontal_speed_m_s",
                "pitch_deg",
                "heading_deg",
                "target_pitch_deg",
                "apoapsis_altitude_m",
                "periapsis_altitude_m",
                "dynamic_pressure_pa",
                "mass_t",
                "thrust_n",
                "throttle",
                "current_stage",
            ]
        )

        control.sas = False
        control.rcs = False
        control.throttle = 1.0
        autopilot.reference_frame = vessel.surface_reference_frame
        autopilot.target_pitch_and_heading(90.0, HEADING)
        autopilot.engaged = True
        autopilot.wait()

        print(
            f"Пуск. Вертикальный участок до "
            f"{VERTICAL_ASCENT_END / 1000.0:.1f} км."
        )
        control.activate_next_stage()

        ascent_apoapsis_target = (
            TARGET_ORBIT_ALTITUDE - APOAPSIS_CUTOFF_MARGIN
        )
        while vessel.orbit.apoapsis_altitude < ascent_apoapsis_target:
            now = conn.space_center.ut
            altitude = flight.mean_altitude
            target_pitch = commanded_pitch(altitude)
            autopilot.target_pitch_and_heading(target_pitch, HEADING)

            # Тяжёлая версия может на несколько десятков секунд пройти первый
            # апоцентр на безопасной высоте около 80 км. Это ещё не авария:
            # продолжающийся почти горизонтальный импульс быстро поднимает
            # перицентр и формирует связанную орбиту. Останавливаемся только
            # при настоящем падении обратно к атмосфере.
            if (
                flight.vertical_speed < MAX_SAFE_DESCENT_SPEED
                and altitude < MIN_SAFE_UPPER_STAGE_ALTITUDE
                and vessel.orbit.periapsis_altitude < 70_000.0
            ):
                control.throttle = 0.0
                raise RuntimeError(
                    "Орбитальная ступень опасно снижается к атмосфере до "
                    "формирования безопасного перицентра."
                )

            throttle = atmospheric_throttle(altitude, flight.dynamic_pressure)
            apoapsis = vessel.orbit.apoapsis_altitude
            if apoapsis > TARGET_ORBIT_ALTITUDE - 15_000.0:
                # Плавное выключение, чтобы не перелететь целевой апоцентр.
                approach_throttle = lerp(
                    apoapsis,
                    TARGET_ORBIT_ALTITUDE - 15_000.0,
                    TARGET_ORBIT_ALTITUDE,
                    1.0,
                    0.15,
                )
                throttle = min(throttle, approach_throttle)
            control.throttle = throttle

            vessel, vessel_changed = stage_if_needed(
                conn, vessel, state, booster_parts
            )
            if vessel_changed:
                control = vessel.control
                autopilot = vessel.auto_pilot
                flight = vessel.flight(vessel.orbit.body.reference_frame)
                control.sas = False
                control.rcs = False
                control.throttle = throttle
                autopilot.reference_frame = vessel.surface_reference_frame
                autopilot.target_pitch_and_heading(target_pitch, HEADING)
                autopilot.engaged = True

            mission_time = now - start_ut
            if mission_time - last_log_time >= TELEMETRY_PERIOD:
                writer.writerow(
                    [
                        f"{mission_time:.3f}",
                        f"{altitude:.3f}",
                        f"{flight.speed:.3f}",
                        f"{flight.vertical_speed:.3f}",
                        f"{flight.horizontal_speed:.3f}",
                        f"{flight.pitch:.3f}",
                        f"{flight.heading:.3f}",
                        f"{target_pitch:.3f}",
                        f"{apoapsis:.3f}",
                        f"{vessel.orbit.periapsis_altitude:.3f}",
                        f"{flight.dynamic_pressure:.3f}",
                        f"{vessel.mass / 1000.0:.5f}",
                        f"{vessel.thrust:.3f}",
                        f"{control.throttle:.4f}",
                        control.current_stage,
                    ]
                )
                csv_file.flush()
                last_log_time = mission_time

            time.sleep(0.05)

        control.throttle = 0.0
        print(
            f"Апоцентр для циркуляризации получен: "
            f"{vessel.orbit.apoapsis_altitude / 1000:.1f} км "
            f"(допуск {APOAPSIS_CUTOFF_MARGIN / 1000:.1f} км)."
        )

        # Выходим из атмосферы и создаём узел в апоцентре.
        while flight.mean_altitude < 72_000.0:
            time.sleep(0.2)

        delta_v = circularization_delta_v(vessel)
        node_ut = conn.space_center.ut + vessel.orbit.time_to_apoapsis
        node = control.add_node(node_ut, prograde=delta_v)
        vessel = execute_node(conn, vessel, node, state)

    periapsis = vessel.orbit.periapsis_altitude
    apoapsis = vessel.orbit.apoapsis_altitude
    print(
        f"Орбита достигнута: Pe={periapsis / 1000:.1f} км, "
        f"Ap={apoapsis / 1000:.1f} км."
    )
    print(f"Телеметрия сохранена: {log_path}")
    conn.close()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        emergency_stop()
        print("\nАвтопилот остановлен пользователем.")
    except Exception as error:
        emergency_stop()
        print(f"\nАВАРИЙНАЯ ОСТАНОВКА: {error}")
        raise
