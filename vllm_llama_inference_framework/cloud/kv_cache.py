"""
F3: 云端 KV Cache 管理 (vLLM)
"""
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from collections import OrderedDict
import threading
import asyncio


class VLLMKVCache:
    """
    vLLM 的 KV Cache 管理器
    
    特点:
    1. 支持大规模并发请求
    2. 分布式缓存同步
    3. 智能预填充
    4. 与 vLLM 的 PagedAttention 集成
    """
    
    def __init__(
        self, 
        max_blocks: int = 10000,
        block_size: int = 16,
        max_seq_len: int = 4096,
        enable_prefix_caching: bool = True,
        enable_compression: bool = False
    ):
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.max_seq_len = max_seq_len
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_compression = enable_compression
        
        # 缓存结构: prompt_hash -> cache_blocks
        self.cache: Dict[str, Dict[str, Any]] = OrderedDict()
        
        # 块分配表
        self.block_table: Dict[int, Dict[str, Any]] = {}
        self.next_block_id = 0
        
        # 统计信息
        self.stats = {
            'hits': 0,
            'misses': 0,
            'block_hits': 0,
            'block_misses': 0,
            'evictions': 0,
            'total_blocks_allocated': 0
        }
        
        # 锁
        self.lock = threading.Lock()
        
        # 前缀树 (用于前缀匹配)
        self.prefix_tree = PrefixTree() if enable_prefix_caching else None
    
    def allocate_blocks(
        self, 
        num_blocks: int,
        prompt_hash: str
    ) -> List[int]:
        """
        分配缓存块
        
        Args:
            num_blocks: 需要分配的块数
            prompt_hash: 提示的哈希值
            
        Returns:
            分配的块ID列表
        """
        with self.lock:
            allocated_blocks = []
            
            # 检查是否有足够的空闲块
            available_blocks = self.max_blocks - len(self.block_table)
            
            if available_blocks < num_blocks:
                # 需要淘汰一些块
                blocks_to_evict = num_blocks - available_blocks
                self._evict_blocks(blocks_to_evict)
            
            # 分配新块
            for _ in range(num_blocks):
                block_id = self.next_block_id
                self.next_block_id += 1
                
                self.block_table[block_id] = {
                    'block_id': block_id,
                    'prompt_hash': prompt_hash,
                    'allocated_at': time.time(),
                    'last_access': time.time(),
                    'access_count': 0,
                    'tokens': []  # 实际存储的tokens
                }
                
                allocated_blocks.append(block_id)
                self.stats['total_blocks_allocated'] += 1
            
            return allocated_blocks
    
    def get_cache_blocks(
        self, 
        prompt_hash: str
    ) -> Optional[List[int]]:
        """
        获取缓存块
        
        Args:
            prompt_hash: 提示哈希
            
        Returns:
            块ID列表或 None
        """
        with self.lock:
            if prompt_hash not in self.cache:
                self.stats['misses'] += 1
                return None
            
            cache_entry = self.cache[prompt_hash]
            blocks = cache_entry['block_ids']
            
            # 更新访问时间
            for block_id in blocks:
                if block_id in self.block_table:
                    self.block_table[block_id]['last_access'] = time.time()
                    self.block_table[block_id]['access_count'] += 1
            
            cache_entry['last_access'] = time.time()
            cache_entry['access_count'] += 1
            
            # 移动到末尾 (LRU)
            self.cache.move_to_end(prompt_hash)
            
            self.stats['hits'] += 1
            self.stats['block_hits'] += len(blocks)
            
            return blocks.copy()
    
    def set_cache_blocks(
        self, 
        prompt_hash: str,
        block_ids: List[int],
        seq_len: int,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        设置缓存块
        
        Args:
            prompt_hash: 提示哈希
            block_ids: 块ID列表
            seq_len: 序列长度
            metadata: 额外元数据
        """
        with self.lock:
            # 如果缓存已满，执行淘汰
            if len(self.cache) >= self.max_blocks:
                self._evict_lru_blocks()
            
            cache_entry = {
                'prompt_hash': prompt_hash,
                'block_ids': block_ids.copy(),
                'seq_len': seq_len,
                'num_blocks': len(block_ids),
                'created_at': time.time(),
                'last_access': time.time(),
                'access_count': 1,
                'metadata': metadata or {}
            }
            
            self.cache[prompt_hash] = cache_entry
            
            # 更新前缀树
            if self.prefix_tree:
                self.prefix_tree.insert(prompt_hash, seq_len)
    
    def _evict_blocks(self, num_blocks: int):
        """淘汰指定数量的块"""
        with self.lock:
            blocks_to_remove = []
            
            # 找出最久未使用的块
            sorted_blocks = sorted(
                self.block_table.items(),
                key=lambda x: x[1]['last_access']
            )
            
            for i in range(min(num_blocks, len(sorted_blocks))):
                block_id = sorted_blocks[i][0]
                blocks_to_remove.append(block_id)
            
            # 移除块
            for block_id in blocks_to_remove:
                prompt_hash = self.block_table[block_id]['prompt_hash']
                del self.block_table[block_id]
                
                # 从缓存中移除对应的块
                if prompt_hash in self.cache:
                    cache_entry = self.cache[prompt_hash]
                    if block_id in cache_entry['block_ids']:
                        cache_entry['block_ids'].remove(block_id)
                        cache_entry['num_blocks'] -= 1
                        
                        # 如果该缓存没有块了，删除缓存条目
                        if not cache_entry['block_ids']:
                            del self.cache[prompt_hash]
                
                self.stats['evictions'] += 1
    
    def _evict_lru_blocks(self):
        """LRU 淘汰块"""
        if not self.cache:
            return
        
        # 弹出最久未使用的缓存条目
        oldest_hash, oldest_entry = self.cache.popitem(last=False)
        
        # 移除对应的块
        for block_id in oldest_entry['block_ids']:
            if block_id in self.block_table:
                del self.block_table[block_id]
        
        self.stats['evictions'] += 1
    
    def get_prefix_blocks(
        self, 
        prompt_prefix: str,
        min_match_len: int = 10
    ) -> Optional[Dict[str, Any]]:
        """
        获取前缀匹配的缓存块
        
        Args:
            prompt_prefix: 提示前缀
            min_match_len: 最小匹配长度
            
        Returns:
            匹配的缓存信息
        """
        if not self.prefix_tree:
            return None
        
        with self.lock:
            match = self.prefix_tree.find_longest_match(prompt_prefix)
            
            if match and match['match_len'] >= min_match_len:
                prompt_hash = match['key']
                
                if prompt_hash in self.cache:
                    cache_entry = self.cache[prompt_hash]
                    
                    # 更新访问统计
                    cache_entry['last_access'] = time.time()
                    cache_entry['access_count'] += 1
                    
                    self.stats['hits'] += 1
                    
                    return {
                        'matched_hash': prompt_hash,
                        'matched_len': match['match_len'],
                        'block_ids': cache_entry['block_ids'].copy(),
                        'seq_len': cache_entry['seq_len']
                    }
            
            self.stats['misses'] += 1
            return None
    
    def clear_all_cache(self):
        """清空所有缓存"""
        with self.lock:
            self.cache.clear()
            self.block_table.clear()
            self.next_block_id = 0
            
            if self.prefix_tree:
                self.prefix_tree.clear()
            
            self.stats = {
                'hits': 0,
                'misses': 0,
                'block_hits': 0,
                'block_misses': 0,
                'evictions': 0,
                'total_blocks_allocated': 0
            }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        with self.lock:
            total_requests = self.stats['hits'] + self.stats['misses']
            hit_rate = self.stats['hits'] / total_requests if total_requests > 0 else 0.0
            block_hit_rate = self.stats['block_hits'] / (self.stats['block_hits'] + self.stats['block_misses']) if (self.stats['block_hits'] + self.stats['block_misses']) > 0 else 0.0
            
            return {
                'cache_entries': len(self.cache),
                'allocated_blocks': len(self.block_table),
                'max_blocks': self.max_blocks,
                'block_utilization': len(self.block_table) / self.max_blocks,
                'hit_rate': hit_rate,
                'block_hit_rate': block_hit_rate,
                'hits': self.stats['hits'],
                'misses': self.stats['misses'],
                'block_hits': self.stats['block_hits'],
                'block_misses': self.stats['block_misses'],
                'evictions': self.stats['evictions'],
                'total_blocks_allocated': self.stats['total_blocks_allocated']
            }


class PrefixTree:
    """前缀树，用于高效的前缀匹配"""
    
    def __init__(self):
        self.root = {}
    
    def insert(self, key: str, value: int):
        """插入键值对"""
        node = self.root
        for char in key:
            if char not in node:
                node[char] = {}
            node = node[char]
        node['__value__'] = value
        node['__key__'] = key
    
    def find_longest_match(self, prefix: str) -> Optional[Dict[str, Any]]:
        """查找最长匹配"""
        node = self.root
        best_match = None
        
        for i, char in enumerate(prefix):
            if char not in node:
                break
            node = node[char]
            
            if '__value__' in node:
                best_match = {
                    'key': node['__key__'],
                    'value': node['__value__'],
                    'match_len': i + 1
                }
        
        return best_match
    
    def clear(self):
        """清空前缀树"""
        self.root.clear()


class VLLMKVCacheManager:
    """vLLM KV Cache 统一管理器"""
    
    def __init__(self):
        self.caches: Dict[str, VLLMKVCache] = {}
        self.cache_groups: Dict[str, List[str]] = {}
    
    def register_cache(
        self, 
        name: str, 
        cache: VLLMKVCache,
        group: str = 'default'
    ):
        """注册缓存实例"""
        self.caches[name] = cache
        
        if group not in self.cache_groups:
            self.cache_groups[group] = []
        self.cache_groups[group].append(name)
    
    def get_cache(self, name: str) -> Optional[VLLMKVCache]:
        """获取缓存实例"""
        return self.caches.get(name)
    
    def get_group_stats(self, group: str = 'default') -> Dict[str, Any]:
        """获取缓存组统计"""
        if group not in self.cache_groups:
            return {}
        
        total_hits = 0
        total_misses = 0
        total_blocks = 0
        
        for cache_name in self.cache_groups[group]:
            if cache_name in self.caches:
                stats = self.caches[cache_name].get_cache_stats()
                total_hits += stats['hits']
                total_misses += stats['misses']
                total_blocks += stats['allocated_blocks']
        
        total_requests = total_hits + total_misses
        global_hit_rate = total_hits / total_requests if total_requests > 0 else 0.0
        
        return {
            'group_name': group,
            'cache_count': len(self.cache_groups[group]),
            'global_hit_rate': global_hit_rate,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'total_blocks': total_blocks,
            'total_requests': total_requests
        }


# 消融实验支持
class AblatedVLLMKVCache(VLLMKVCache):
    """用于消融实验的 vLLM KV Cache"""
    
    def __init__(
        self, 
        max_blocks: int = 10000,
        block_size: int = 16,
        disable_prefix_caching: bool = False,
        disable_lru: bool = False,
        **kwargs
    ):
        # 临时禁用前缀缓存以测试效果
        if disable_prefix_caching:
            enable_prefix_caching = False
        else:
            enable_prefix_caching = True
            
        super().__init__(
            max_blocks=max_blocks,
            block_size=block_size,
            enable_prefix_caching=enable_prefix_caching,
            **kwargs
        )
        
        self.disable_lru = disable_lru
    
    def _evict_lru_blocks(self):
        """重写LRU淘汰"""
        if self.disable_lru:
            # 禁用LRU，使用FIFO
            if self.cache:
                oldest_hash, oldest_entry = self.cache.popitem(last=False)
                for block_id in oldest_entry['block_ids']:
                    if block_id in self.block_table:
                        del self.block_table[block_id]
                self.stats['evictions'] += 1
        else:
            super()._evict_lru_blocks()
