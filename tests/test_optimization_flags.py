from click.testing import CliRunner
import yaml
from distillkit.configuration import DistillationRunConfig
from distillkit.main import main


def test_optimization_flags_defaults(tmp_path):
    raw_config = {
        'model': 'dummy',
        'dataset': {},
        'teacher': {'kind': 'dataset', 'cache_path': 'dummy'},
        'sequence_length': 128,
        'output_path': str(tmp_path / 'out'),
    }
    cfg = DistillationRunConfig.model_validate(raw_config)
    assert cfg.allow_tf32 is True
    assert cfg.cuda_allocator_gc_threshold == 0.8
    assert cfg.high_resolution_timer is True


def test_optimization_flags_config_override(tmp_path):
    raw_config = {
        'model': 'dummy',
        'dataset': {},
        'teacher': {'kind': 'dataset', 'cache_path': 'dummy'},
        'sequence_length': 128,
        'output_path': str(tmp_path / 'out'),
        'allow_tf32': False,
        'cuda_allocator_gc_threshold': 0.7,
        'high_resolution_timer': False,
    }
    cfg = DistillationRunConfig.model_validate(raw_config)
    assert cfg.allow_tf32 is False
    assert cfg.cuda_allocator_gc_threshold == 0.7
    assert cfg.high_resolution_timer is False


def test_cli_overrides_optimization_flags(tmp_path, monkeypatch):
    config_data = {
        'model': 'dummy',
        'dataset': {},
        'teacher': {'kind': 'dataset', 'cache_path': 'dummy'},
        'sequence_length': 128,
        'output_path': str(tmp_path / 'out'),
        'allow_tf32': True,
        'cuda_allocator_gc_threshold': 0.8,
        'high_resolution_timer': True,
        'max_vram_fraction': 0.95,
    }
    config_file = tmp_path / 'test_config.yml'
    config_file.write_text(yaml.dump(config_data))

    captured_config = []

    def mock_do_distill(cfg, initialise_only=False):
        captured_config.append(cfg)

    monkeypatch.setattr('distillkit.main.do_distill', mock_do_distill)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            str(config_file),
            '--no-allow-tf32',
            '--cuda-allocator-gc-threshold',
            '0.6',
            '--no-high-resolution-timer',
            '--max-vram-fraction',
            '0.90',
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(captured_config) == 1
    cfg = captured_config[0]
    assert cfg.allow_tf32 is False
    assert cfg.cuda_allocator_gc_threshold == 0.6
    assert cfg.high_resolution_timer is False
    assert cfg.max_vram_fraction == 0.90
