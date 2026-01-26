"""
F3: 边端 KV Cache 管理 (llama.cpp)
"""
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from collections import OrderedDict
import threading


class LlamaCppKVCache:
    """
    llama.cpp 的 KV Cache 管理器
    
    特点:
    1. 支持多轮对话缓存
    2. LRU 淘汰策略
    3. 缓存预热和预分配
    4. 统计和监控
    """
    
    def __init__(
        self, 
        max_size: int = 1000,
        max_seq_len: int = 2048,
        enable_compression: bool = False
    ):
        self.max_size = max_size
        self.max_seq_len = max_seq_len
        self.enable_compression = enable_compression
        
        # LRU Cache: prompt -> cache_data
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        
        # 统计信息
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'total_tokens_cached': 0
        }
        
        # 锁，用于线程安全
        self.lock = threading.Lock()
    
    def get_cache(
        self, 
        prompt: str,
        max_tokens: int = 64
    ) -> Optional[Dict[str, Any]]:
        """
        从缓存获取 KV 值
        
        Args:
            prompt: 输入提示
            max_tokens: 预期生成的最大token数
            
        Returns:
            缓存数据或 None
        """
        with self.lock:
            if prompt not in self.cache:
                self.stats['misses'] += 1
                return None
            
            # 移动到末尾 (LRU)
            self.cache.move_to_end(prompt)
            
            cache_data = self.cache[prompt]
            
            # 检查缓存是否足够
            available_tokens = cache_data.get('available_tokens', 0)
            if available_tokens < max_tokens:
                # 缓存不足以生成所需token
                self.stats['misses'] += 1
                return None
            
            self.stats['hits'] += 1
            cache_data['last_access'] = time.time()
            
            return cache_data.copy()
    
    def set_cache(
        self, 
        prompt: str,
        token_ids: List[int],
        kv_tensors: Optional[Dict[str, Any]] = None,
        available_tokens: int = 0
    ):
        """
        设置缓存
        
        Args:
            prompt: 输入提示
            token_ids: 已生成的token ID列表
            kv_tensors: KV tensor数据 (实际llama.cpp的数据)
            available_tokens: 可用于继续生成的token数
        """
        with self.lock:
            # 如果缓存已满，执行LRU淘汰
            if len(self.cache) >= self.max_size:
                self._evict_lru()
            
            cache_data = {
                'prompt': prompt,
                'token_ids': token_ids.copy(),
                'seq_len': len(token_ids),
                'kv_tensors': kv_tensors,
                'available_tokens': available_tokens,
                'created_at': time.time(),
                'last_access': time.time(),
                'access_count': 1
            }
            
            self.cache[prompt] = cache_data
            self.stats['total_tokens_cached'] += len(token_ids)
    
    def _evict_lru(self):
        """LRU 淘汰策略"""
        if not self.cache:
            return
        
        # 弹出最久未使用的项
        oldest_key, oldest_value = self.cache.popitem(last=False)
        self.stats['evictions'] += 1
        self.stats['total_tokens_cached'] -= len(oldest_value.get('token_ids', []))
    
    def update_cache_availability(
        self, 
        prompt: str,
        new_available_tokens: int
    ):
        """更新缓存的可用token数"""
        with self.lock:
            if prompt in self.cache:
                self.cache[prompt]['available_tokens'] = new_available_tokens
                self.cache[prompt]['last_access'] = time.time()
    
    def get_partial_cache(
        self, 
        prompt: str,
        prefix_len: int
    ) -> Optional[Dict[str, Any]]:
        """
        获取部分缓存 (用于前缀匹配)
        
        Args:
            prompt: 完整prompt
            prefix_len: 需要匹配的前缀长度
            
        Returns:
            部分缓存数据
        """
        with self.lock:
            # 查找最长前缀匹配
            best_match = None
            best_match_len = 0
            
            for cached_prompt in self.cache.keys():
                if prompt.startswith(cached_prompt):
                    match_len = len(cached_prompt)
                    if match_len > best_match_len:
                        best_match = cached_prompt
                        best_match_len = match_len
            
            if best_match and best_match_len >= prefix_len:
                self.stats['hits'] += 1
                cache_data = self.cache[best_match]
                self.cache.move_to_end(best_match)
                
                return {
                    'matched_prompt': best_match,
                    'matched_len': best_match_len,
                    'cache_data': cache_data.copy()
                }
            
            self.stats['misses'] += 1
            return None
    
    def clear_cache(self):
        """清空缓存"""
        with self.lock:
            self.cache.clear()
            self.stats = {
                'hits': 0,
                'misses': 0,
                'evictions': 0,
                'total_tokens_cached': 0
            }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0.0
            
            return {
                'cache_size': len(self.cache),
                'cache_capacity': self.max_size,
                'hit_rate': hit_rate,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'evictions': self.stats['evictions'],
                'total_tokens_cached': self.stats['total_tokens_cached'],
                'avg_seq_len': self.stats['total_tokens_cached'] / len(self.cache) if self.cache else 0
            }
    
    def export_cache(self, filepath: str):
        """导出缓存到文件"""
        with self.lock:
            cache_data = {
                'cache': {k: {
                    'prompt': v['prompt'],
                    'token_ids': v['token_ids'],
                    'seq_len': v['seq_len'],
                    'available_tokens': v['available_tokens'],
                    'created_at': v['created_at'],
                    'last_access': v['last_access'],
                    'access_count': v['access_count']
                } for k, v in self.cache.items()},
                'stats': self.stats
            }
            
            with open(filepath, 'w') as f:
                json.dump(cache_data, f, indent=2)
    
    def import_cache(self, filepath: str):
        """从文件导入缓存"""
        with self.lock:
            with open(filepath, 'r') as f:
                cache_data = json.load(f)
            
            # 导入统计数据
            self.stats.update(cache_data.get('stats', {}))
            
            # 导入缓存数据
            for prompt, data in cache_data.get('cache', {}).items():
                self.cache[prompt] = {
                    'prompt': data['prompt'],
                    'token_ids': data['token_ids'],
                    'seq_len': data['seq_len'],
                    'available_tokens': data['available_tokens'],
                    'created_at': data['created_at'],
                    'last_access': data['last_access'],
                    'access_count': data['access_count']
                }


class KVCacheManager:
    """KV Cache 统一管理器"""
    
    def __init__(self):
        self.caches: Dict[str, LlamaCppKVCache] = {}
    
    def register_cache(
        self, 
        name: str, 
        cache: LlamaCppKVCache
    ):
        """注册缓存实例"""
        self.caches[name] = cache
    
    def get_cache(self, name: str) -> Optional[LlamaCppKVCache]:
        """获取缓存实例"""
        return self.caches.get(name)
    
    def get_global_stats(self) -> Dict[str, Any]:
        """获取全局统计"""
        total_hits = 0
        total_misses = 0
        total_caches = 0
        
        for name, cache in self.caches.items():
            stats = cache.get_cache_stats()
            total_hits += stats['hits']
            total_misses += stats['misses']
            total_caches += 1
        
        total_requests = total_hits + total_misses
        global_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'total_caches': total_caches,
            'global_hit_rate': global_hit_rate,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'total_requests': total_requests
        }


# 消融实验支持
class AblatedKVCache(LlamaCppKVCache):
    """用于消融实验的 KV Cache（禁用特定功能）"""
    
    def __init__(
        self, 
        max_size: int = 1000,
        max_seq_len: int = 2048,
        disable_lru: bool = False,
        disable_prefix_match: bool = False,
        **kwargs
    ):
        super().__init__(max_size, max_seq_len, **kwargs)
        self.disable_lru = disable_lru
        self.disable_prefix_match = disable_prefix_match
    
    def _evict_lru(self):
        """重写LRU淘汰"""
        if self.disable_lru:
            # 禁用LRU，使用随机淘汰
            if self.cache:
                random_key = list(self.cache.keys())[0]
                del self.cache[random_key]
                self.stats['evictions'] += 1
        else:
            super()._evict_lru()
    
    def get_partial_cache(self, prompt: str, prefix_len: int):
        """重写前缀匹配"""
        if self.disable_prefix_match:
            # 禁用前缀匹配，只支持完全匹配
            return None
        return super().get_partial_cache(prompt, prefix_len)
